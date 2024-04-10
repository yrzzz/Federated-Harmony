import numpy as np
import scipy as sc
import scipy.sparse.linalg as lsa
#import python.PCA.comparison as co



def svd_sub(data, ndims):
    # the sparse matrix version of svd has better memory requirements, while being a little
    # slower
    # covariance matrix is positive semi definite so SVD= Eigenvalue decomposition
    # print(nd)
    nd = min(min(data.shape[1] - 1, data.shape[0] - 1), ndims)
    #print(nd)
    u, s, v = lsa.svds(data, k=nd)
    # Sparse returns eigenvalues in ascending order
    s = np.flip(s)
    v = np.flip(v.T, axis=1)
    u = np.flip(u, axis=1)
    return u,s,v, nd


def compute_cov(original):
    """
    Compute a covariance matrix of a matrix
    :return: the noisy covariance matrix
    """
    # compute covariance matrix
    # n = number of samples
    n = original.shape[0]
    # np.dot is matrix product for 2 dimensional arrays
    cov = (1 / (n - 1)) * np.dot(original.transpose(), original)
    return cov


def variance_explained(eigenvalues, perc=0.5):
    total_variance = sum(eigenvalues)
    percentages = eigenvalues / total_variance
    p = 0
    sum_perc = 0
    while sum_perc < perc and p < len(eigenvalues):
        sum_perc = sum_perc + percentages[p]
        p = p + 1
    return p


def generate_random_gaussian(n, m):
    draws = n * m
    noise = np.random.normal(0, 1, draws)
    #print('Generated random initial matrix: finished sampling')
    # make a matrix out of the noise
    noise.shape = (n, m)
    # transform matrix into s
    return noise


def extract_eigenvals(eigenvalues):
    """
    eigenvaluesigendecomposition from scipy.linalg.sparse returns eigenvalues ordered in
    increasing order, followed by eigenvalues which are 0.
    Eigenvalues are returned in decreasing order ommiting th 0s alltogether
    Args:
        eigenvalues: Eigenvalues from a sparse singular value decomposition.

    Returns: Eigenvalue vector in decreasing order, without 0s.

    """
    indz = np.where(eigenvalues == 0)
    eigenvalues = np.flip(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues != 0]
    return eigenvalues, indz


def projection(scaled, sim, ndims):
    projection = sc.dot(scaled, sim[:, 0:ndims])
    return projection


def partition_data_horizontally(data, splits=2, equal=True, randomize=False, perc = []):
    """

    Split a numpy matrix into horizontal chunks for testing.

    Args:
        data: The entire data set formatted as a numpy matrix
        splits: The number of chunks to split the data into
        equal: If true, equal chunks of data are returned
        randomize: Randomize the data before splitting it
        perc: if equal is False and a vector of percentages is provided, the data is
                split into fractions accroding to this vector.

    Returns: A list of numpy arrays with the horizontal chunks of data.

    """
    n = data.shape[0]
    interval_end = []
    interval_end = []
    if equal:
        for s in range(splits - 1):
            interval_end.append(int(np.floor((s + 1) * n / splits)))
        # last one should include all the elements!
        interval_end.append(n)
    else:
        psum = 0
        for p in perc:
            psum = psum + p
            interval_end.append(int(np.floor(psum * n)))
        # last one should include all the remaining elements!
        interval_end.append(n)

    if randomize:
        # generate a permutation of the indices and return the permuted data
        choice = np.random.choice(n, n, replace=False)
        data = data[choice, :]
    else:
        choice = range(data.shape[0])
    start = 0
    localdata = []
    for i in range(len(interval_end)):
        end = int(interval_end[i])
        # slice matrix
        data_sub = data[start:end, :]
        # calculate covariance matrix
        start = int(interval_end[i])
        localdata.append(data_sub)
    return localdata, choice


def partition_data_vertically(data, splits=2, equal=True, randomize=False, perc=[]):
    """

    Split a numpy matrix into vertical chunks for testing.

    Args:
        data: The entire data set formatted as a numpy matrix
        splits: The number of chunks to split the data into
        equal: If true, equal chunks of data are returned
        randomize: Randomize the data before splitting it
        perc: if equal is False and a vector of percentages is provided, the data is
                split into fractions accroding to this vector.

    Returns: A list of numpy arrays with the vertical chunks of data.

    """
    n = data.shape[1]
    # save the end of the interval
    # with 100 samples and 3 sites:[33,66,100]
    interval_end = []
    if equal:
        for s in range(splits - 1):
            interval_end.append(int(np.floor((s + 1) * n / splits)))
        # last one should include all the elements!
        interval_end.append(n)
    else:
        psum = 0
        for p in perc:
            psum = psum+p
            interval_end.append(int(np.floor(psum*n)))
        # last one should include all the remaining elements!
        interval_end.append(n)
    if randomize:
        # generate a permutation of the indices and return the permuted data
        choice = np.random.choice(n, n, replace=False)
        data = data[:, choice]
    else:
        choice = range(n)

    start = 0
    localdata = []
    for i in range(len(interval_end)):
        end = int(interval_end[i])
        # slice matrix
        data_sub = data[:, start:end]
        start = int(interval_end[i])
        localdata.append(data_sub)
    return localdata, choice


def eigenvalue(A, v):
    """
    Computes a single eigenvector based on a matrix and a corresponding
    eigenvector
    Args:
        A: the matrix
        v: the eigenvector

    Returns: The eigenvalue

    """

    Av = A.dot(v)
    return v.dot(Av)


def eigenvalues(eigenvectors, cov):
    """
    Computes eigenvalues based on a covariance matrix and the corresponding
    eigenvectors
    Args:
        eigenvectors: Eigenvectors of the covariance matrix
        cov: Covariance matrix

    Returns: All eigenvalues corresponding to the eigenvectors

    """

    eigenvals = []
    for v in range(eigenvectors.shape[1]):
        eigenvals.append(eigenvalue(cov, eigenvectors[:, v]))
    return eigenvals

def eigenvector_convergence_checker(current, previous, tolerance=1e-9, required=None):
    '''

    This function checks whether two sets of vectors are assymptotically collinear, up
    to a tolerance of epsilon.
    Args:
        current: The current eigenvector estimate
        previous: The eigenvector estimate from the previous iteration
        tolerance: The error tolerance for eigenvectors to be equal
        required: optional parameter for the number of eigenvectors required to have converged

    Returns: True if the required numbers of eigenvectors have converged to the given precision, False otherwise
                deltas, the current difference between the dot products

    '''
    nr_converged = 0
    col = 0
    converged = False
    deltas = []
    if required is None:
        required = current.shape[1]
    while col < current.shape[1] and not converged:
        # check if the scalar product of the current and the previous eigenvectors
        # is 1, which means the vectors are 'parallel'
        delta = np.abs(np.sum(np.dot(np.transpose(current[:, col]), previous[:, col])))
        deltas.append(delta)
        if delta >= 1 - tolerance:
            nr_converged = nr_converged + 1
        if nr_converged >= required:
            converged = True
        col = col + 1
    return converged, deltas

def convergence_checker_rayleigh(current, previous, alpha_current, alpha_prev, epsilon=1e-11, required=None):
    """
    Convergence checked via Raleigh coefficient. which is the dot product of two consecutive eigenvectors
    divided by the norm.
    This function here implements the Rayleigh coefficient according to the definition in
    'A covariance free iterative algorithm for principal component analysis' (Guo, 2012).
    Args:
        current: Current eigenvector estimate (H)
        previous: Previous iterations' eigenvector estimate
        alpha_current: Current norm of G
        alpha_prev: previous norm of G
        epsilon: tolerance ( Default value arbitrarily chosen, to approximate the same behaviour as
                    angle based convergence checker)

    Returns: True if converged otherwise false, deltas the differences rayleigh coefficients

    """

    nr_converged = 0
    col = 0
    converged = False
    deltas = []
    if required is None:
        required = current.shape[1]
    while col < current.shape[1] and not converged:
        ra = np.dot(current[:,col].T, current[:,col])/ alpha_current[col]
        rap = np.dot(previous[:,col].T, previous[:,col])/ alpha_prev[col]
        deltas.append(np.abs(ra-rap))
        if np.abs(ra-rap) < epsilon:
            nr_converged = nr_converged+1
        if nr_converged >= required:
            converged = True
        col = col + 1

    return converged, deltas


