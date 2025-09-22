import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('harmonypy')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Center:
    def __init__(self, Y, client_list, epsilon_cluster=1e-5, epsilon_harmony=1e-4, sigma=0.1, tau=0, block_size=0.05, nclust=None, theta=None, lamb=None):
        """

        :param Y: Centorids after FedKmeans, with dimension d×K
        :param client_list:
        :param sigma: float
        :param nclust: number of initial cluster

        """
        self.Y = Y / np.linalg.norm(Y, ord=2, axis=0)
        self.client_list = client_list
        self.Ni_list = [client.Ni for client in self.client_list]
        self.N = sum(self.Ni_list)
        self.block_size = block_size
        self.window_size = 3
        self.epsilon_cluster = epsilon_cluster
        self.epsilon_harmony = epsilon_harmony
        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []
        total_columns = [client.Z_cos.shape[1] for client in client_list]
        self.cumulative_columns = np.cumsum([0] + total_columns)
        phi = np.zeros((len(self.client_list), self.N), dtype=bool)
        phi_n = np.array([len(client_list)])
        # Fill in the indicator matrix
        current_column = 0
        for i, client in enumerate(self.client_list):
            num_cols = client.Ni
            phi[i, current_column:current_column + num_cols] = True
            current_column += num_cols
        self.phi = phi
        self.phi_moe = np.vstack((np.repeat(1, self.N), phi))
        N_b = self.phi.sum(axis=1)
        # Proportion of items in each category.
        self.Pr_b = N_b / self.N

        if nclust is None:
            self.nclust = np.min([np.round(self.N / 30.0), 100]).astype(int)

        if type(sigma) is float and self.nclust > 1:
            self.sigma = np.repeat(sigma, self.nclust)
        else:
            self.sigma = sigma

        if theta is None:
            self.theta = np.repeat([1] * len(phi_n), phi_n)
        elif isinstance(theta, float) or isinstance(theta, int):
            self.theta = np.repeat([theta] * len(phi_n), phi_n)
        elif len(theta) == len(phi_n):
            self.theta = np.repeat([theta], phi_n)

        assert len(self.theta) == np.sum(phi_n), \
            "each batch variable must have a theta"

        if tau > 0:
            self.theta = self.theta * (1 - np.exp(-(N_b / (self.nclust * tau)) ** 2))

        if lamb is None:
            lamb = np.repeat([1] * len(phi_n), phi_n)
        elif isinstance(lamb, float) or isinstance(lamb, int):
            lamb = np.repeat([lamb] * len(phi_n), phi_n)
        elif len(lamb) == len(phi_n):
            lamb = np.repeat([lamb], phi_n)

        assert len(lamb) == np.sum(phi_n), \
            "each batch variable must have a lambda"

        self.lamb_mat = np.diag(np.insert(lamb, 0, 0))
    def init_cluster(self):
        rowsum_Ri_list = [client.rowsum_Ri for client in self.client_list]
        self.O = np.column_stack(rowsum_Ri_list)
        self.E = np.outer(np.sum(self.O, axis=1), self.Pr_b)


    def compute_objective(self):
        kmeans_error = sum(client.kmeans_error_i for client in self.client_list)
        _entropy = sum(client._entropy_i for client in self.client_list)
        _cross_entropy = sum(client._cross_entropy_i for client in self.client_list)
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)
    def save_result(self):
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_Y(self):
        self.Y = sum(client.Yi for client in self.client_list)
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)

    def update_R(self):
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        self.blocks = np.array_split(update_order, n_blocks)

    def remove_cells(self):
        rowsum_Rb_list = [client.rowsum_Rb for client in self.client_list]
        self.E -= np.outer(sum(rowsum_Rb_list), self.Pr_b)
        self.O -= np.column_stack(rowsum_Rb_list)


    def put_cells_back(self):
        rowsum_Rb_list = [np.sum(client.Ri[:, client.local_index_list], axis=1) for client in self.client_list]
        self.E += np.outer(sum(rowsum_Rb_list), self.Pr_b)
        self.O += np.column_stack(rowsum_Rb_list)

    def check_convergence(self, i_type):
        obj_old = 0.0
        obj_new = 0.0
        # Clustering, compute new window mean
        if i_type == 0:
            okl = len(self.objective_kmeans)
            for i in range(self.window_size):
                obj_old += self.objective_kmeans[okl - 2 - i]
                obj_new += self.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_cluster:
                return True
            return False
        # Harmony
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony:
                return True
            return False
        return True

    def moe_correct_ridge(self):
        rowsum_Phi_Rk_i_list = [client.rowsum_Phi_Rk_i for client in self.client_list]
        Phi_Rk_i_sum = np.sum(rowsum_Phi_Rk_i_list)
        # Initialize the matrix with zeros, with a size based on the desired output
        x_size = len(rowsum_Phi_Rk_i_list) + 1  # Adjust based on your requirements
        self.x = np.zeros((x_size, x_size))
        # Set the [0, 0] element to be the sum of the array
        self.x[0, 0] = Phi_Rk_i_sum
        # Fill the first row (starting from the second column) and the first column (starting from the second row) with the array values
        self.x[0, 1:] = rowsum_Phi_Rk_i_list
        self.x[1:, 0] = rowsum_Phi_Rk_i_list
        # Fill in the diagonal with the array elements
        np.fill_diagonal(self.x[1:, 1:], rowsum_Phi_Rk_i_list)
        self.x = self.x + self.lamb_mat

    def calculate_W(self):
        self.W = sum(client.Wi for client in self.client_list)
        self.W[0,:] = 0 # do not remove the intercept


class Client:
    def __init__(self, Zi: pd.DataFrame, index = None):
        """

        :param Zi: At first edition, we directly input Z_cos.
        (will be modified to client data after Fed PCA, with dimension: d×N, corresponding to Z_i)
        :param center: Center class in FedHarmony
        :param Yi:
        """

        self.Z_corr = np.array(Zi)
        self.Z_orig = np.array(Zi)
        self.Z_cos = self.Z_orig / self.Z_orig.max(axis=0)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        # self.Z_cos = Zi
        self.index = index
        self.Ni = Zi.shape[1]
    def init_cluster(self, center):
        self.Y = center.Y # already normalized
        self.sigma = center.sigma

        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.Ri = -self.dist_mat
        self.Ri = self.Ri / self.sigma[:, None]
        self.Ri -= np.max(self.Ri, axis=0)
        self.Ri = np.exp(self.Ri)
        self.Ri = self.Ri / np.sum(self.Ri, axis=0)  # Center has no way to know the sum of Ri
        self.rowsum_Ri = np.sum(self.Ri, axis=1)

    def compute_objective(self, center):
        self.kmeans_error_i = np.sum(np.multiply(self.Ri, self.dist_mat))
        self._entropy_i = np.sum(safe_entropy(self.Ri) * self.sigma[:, np.newaxis])
        x_i = (self.Ri * self.sigma[:, np.newaxis])
        y = np.tile(center.theta[:, np.newaxis], center.nclust).T
        z = np.log((center.O + 1) / (center.E + 1))
        self._cross_entropy_i = np.sum(x_i*((y*z)[:, self.index]).reshape(-1, 1))

    def update_dist_mat(self):
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))

    def update_Yi(self):
        self.Yi = np.dot(self.Z_cos, self.Ri.T)

    def receive_updated_Y(self, center):
        self.Y = center.Y

    def update_R(self):
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma[:, None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)

    def calculate_rowsum_block(self, center, b):
        self.local_index_list = []
        mapped_indices = [find_dataset_and_column(index, center.cumulative_columns) for index in b]
        for dataset_index, local_index in mapped_indices:
            if dataset_index == self.index:
                self.local_index_list.append(local_index)
        self.Rb = (self.Ri[:, self.local_index_list])
        self.rowsum_Rb = np.sum(self.Rb, axis=1)

    def recompute_R_for_removed_cells(self, center):
        self.Ri[:, self.local_index_list] = self._scale_dist[:,self.local_index_list]
        self.Ri[:, self.local_index_list] = np.multiply(
            self.Ri[:, self.local_index_list],
            np.tile(np.power((center.E + 1) / (center.O + 1), center.theta)[:, self.index],
                    (len(self.local_index_list), 1)).T
        )
        self.Ri[:, self.local_index_list] = self.Ri[:, self.local_index_list] / np.linalg.norm(self.Ri[:, self.local_index_list], ord=1, axis=0)

    def calculate_Phi_Rk_i(self, center, i):
        if self.index == 0:
            phi_moe_i = center.phi_moe[:, 0: center.Ni_list[self.index]]
        elif self.index == len(center.client_list) - 1:
            phi_moe_i = center.phi_moe[:, sum(center.Ni_list[:self.index]):]
        else:
            phi_moe_i = center.phi_moe[:, sum(center.Ni_list[:self.index]):sum(center.Ni_list[:self.index+1])]
        self.Phi_Rk_i = np.multiply(phi_moe_i, self.Ri[i, :])
        self.rowsum_Phi_Rk_i = np.sum(self.Phi_Rk_i[0, :])

    def moe_correct_ridge(self, center):
        self.Wi = np.dot(np.dot(np.linalg.inv(center.x), self.Phi_Rk_i), self.Z_orig.T)

    def redefine_Z_corr(self):
        self.Z_corr = self.Z_orig.copy()
    def correct(self, center):
        self.Z_corr -= np.dot(center.W.T, self.Phi_Rk_i)
    def update_Z_cos(self):
        self.Z_cos = self.Z_corr / np.linalg.norm(self.Z_corr, ord=2, axis=0)

def safe_entropy(x: np.array):
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y

def find_dataset_and_column(global_index, cumulative_columns):
    for i, max_col in enumerate(cumulative_columns[1:]):
        if global_index < max_col:
            local_index = global_index - cumulative_columns[i]
            return i, local_index
    raise IndexError("Global index out of bounds")


def FL_harmonize(client_list, center, iter_harmony = 20, max_iter_kmeans = 20, verbose = True):
    '''Init_cluster'''
    for client in client_list:
        client.init_cluster(center)
    center.init_cluster()
    for client in client_list:
        client.compute_objective(center)
    center.compute_objective()
    center.save_result()
    '''Harmonize'''
    converged = False
    for i in range(1, iter_harmony+1):
        if verbose:
            logger.info("Iteration {} of {}".format(i, iter_harmony))
        '''Cluster'''
        for client in client_list:
            client.update_dist_mat()

        for k in range(max_iter_kmeans):
            for client in client_list:
                client.update_Yi()
            center.update_Y()

            for client in client_list:
                client.receive_updated_Y(center)
                client.update_dist_mat()

            '''Update R '''
            for client in client_list:
                client.update_R()

            center.update_R()
            for b in center.blocks:
                for client in client_list:
                    client.calculate_rowsum_block(center, b)

                center.remove_cells()

                for client in client_list:
                    client.recompute_R_for_removed_cells(center)
                center.put_cells_back()
            for client in client_list:
                client.compute_objective(center)
            center.compute_objective()
            if i > center.window_size:
                converged = center.check_convergence(0)
                if converged:
                    break
        center.kmeans_rounds.append(k)
        center.save_result()
        '''Regress out covariates'''
        '''correct'''
        for client in client_list:
            client.redefine_Z_corr()
        for n in range(center.nclust):
            for client in client_list:
                client.calculate_Phi_Rk_i(center, n)
            center.moe_correct_ridge()

            for client in client_list:
                client.moe_correct_ridge(center)
            center.calculate_W()
            for client in client_list:
                client.correct(center)
        for client in client_list:
            client.update_Z_cos()

        converged = center.check_convergence(1)
        if converged:
            if verbose:
                logger.info(
                    "Converged after {} iteration{}"
                    .format(i, 's' if i > 1 else '')
                )
            break
    if verbose and not converged:
        logger.info("Stopped before convergence")

    return 0

if __name__ == "__main__":
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Step by step comparision
    ## import data

    dat_FedPCA = pd.read_csv('/Users/ry/Desktop/Pitt/FL/FedHarmony/data/293tdata_FedPCA.csv', index_col=0).T

    # N×d
    meta_data_9000 = pd.read_csv('/Users/ry/Desktop/Pitt/FL/FedAvg/metadata.csv')
    # N×6
    centroids_central = pd.read_csv('/Users/ry/Desktop/Pitt/FL/FedHarmony/data/centroids_central.csv', index_col=0)
    # k×d
    vars_use = ['sample']

    # Z_corr = np.array(dat_FedPCA)
    # Z_orig = np.array(dat_FedPCA)
    #
    # Z_cos = Z_orig / Z_orig.max(axis=0)
    # Z_cos = Z_cos / np.linalg.norm(Z_cos, ord=2, axis=0)

    clustered_data = {}

    for cluster in meta_data_9000['sample'].unique():
        # Use boolean indexing to select rows where the cluster matches
        clustered_data[cluster] = np.array(dat_FedPCA).T[meta_data_9000[meta_data_9000['sample'] == cluster].index]

    batch = '293t'
    Z_293t = clustered_data[batch].T
    Z_jurkat = clustered_data['jurkat'].T
    Z_half = clustered_data['jurkat_293t_half'].T
    Y = centroids_central.T  # centroids d×k

    "FL version"
    # 1. define client class
    client_293t = Client(Zi=Z_293t)
    client_jurkat = Client(Zi=Z_jurkat)
    client_half = Client(Zi=Z_half)
    client_list = [client_293t, client_jurkat, client_half]
    for i in range(len(client_list)):
        client_list[i].index = i

    # 2. define center class
    center = Center(Y, client_list)
    dat_FedPCA = pd.read_csv('/Users/ry/Desktop/Pitt/FL/FedHarmony/data/293tdata_FedPCA.csv', index_col=0).T
    # d×N
    meta_data_9000 = pd.read_csv('/Users/ry/Desktop/Pitt/FL/FedAvg/metadata.csv')
    # N×6
    centroids_central = pd.read_csv('/Users/ry/Desktop/Pitt/FL/FedHarmony/data/centroids_central.csv', index_col=0)
    # k×d
    vars_use = ['sample']

    clustered_data = {}

    for cluster in meta_data_9000['sample'].unique():
        # Use boolean indexing to select rows where the cluster matches
        clustered_data[cluster] = np.array(dat_FedPCA).T[meta_data_9000[meta_data_9000['sample'] == cluster].index]

    Z_293t = clustered_data['293t'].T
    Z_jurkat = clustered_data['jurkat'].T
    Z_half = clustered_data['jurkat_293t_half'].T
    Y = centroids_central.T # centroids d×k


    "FL version"
    # 1. define client class
    client_293t = Client(Zi=Z_293t)
    client_jurkat = Client(Zi=Z_jurkat)
    client_half = Client(Zi=Z_half)
    client_list = [client_293t, client_jurkat, client_half]
    for i in range(len(client_list)):
        client_list[i].index = i

    # 2. define center class
    center = Center(Y, client_list)


    FL_harmonize(client_list, center, 20, 20)

    res = pd.DataFrame(np.concatenate([client_293t.Z_corr, client_jurkat.Z_corr, client_half.Z_corr], axis = 1))
    res.columns = ['X{}'.format(i + 1) for i in range(res.shape[1])]

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(res.T)

    ####
    # plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=2,
    #             c=[sns.color_palette()[x] for x in metadata['orig.ident'].map({"293t":0, "jurkat":1, "jurkat_293t_half":2})])
    # plt.gca().set_aspect('equal', 'datalim')
    #plt.figure(figsize=(8, 8))
    groups = meta_data_9000['sample'].unique()

    # Color mapping
    colors = {"293t": sns.color_palette()[0], "jurkat": sns.color_palette()[1], "jurkat_293t_half": sns.color_palette()[2]}

    # Loop over each group and plot it separately so we can label them
    for group in groups:
        mask = meta_data_9000['sample'] == group
        plt.scatter(embedding[mask, 0], embedding[mask, 1], alpha=0.2, s=3, c=colors[group], label=group)

    plt.gca().set_aspect('equal', 'datalim')
    plt.legend(fontsize=20)
    plt.title('UMAP of PCA Before Harmony Result', fontsize = 20)
    plt.xlabel('UMAP_1', fontsize = 20)
    plt.ylabel('UMAP_2', fontsize = 20)
    plt.show()





















