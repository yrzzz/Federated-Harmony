import shared_functions as sh
import scipy.sparse.linalg as lsa
import scipy.linalg as la
import scipy as sc
import numpy as np
#from python.PCA.logging import *


def simulate_distributed_horizontal(local, p=20, maxit=500, epsilon=1e-9):
    """
    Simulate a distributed subspace iteration on a list of
    covariance matrices
    Args:
        local: list of data matrices variable X samples
        p: number of eigenvectors
        tolerance: Error tolerance for convergence criterion

    Returns: The eigenvectors, eigenvalues and the number of iterations until convergence

    """
    d = local[0].shape[1]
    X = sh.generate_random_gaussian(d, p)
    X, R = la.qr(X, mode='economic')
    X_prev = X
    # order eigenvectors from largest to smallest, to achive
    # ordered eigenvectors
    converged = False
    count = 0
    while not converged and count < maxit:
        count = count + 1
        # eigenvalues are the column norms of the unnormalised matrix
        X = np.zeros(X.shape)
        for l in local:
            #  are in colums.
            d1 = np.dot(l, X_prev)
            d1 = np.dot(l.T, d1)
            X = X + d1
        E = la.norm(X, axis=0)
        X, R = la.qr(X, mode='economic')
        converged, deltas = sh.eigenvector_convergence_checker(X, X_prev, tolerance=epsilon)
        #log_current_accuracy(v=v, H_i=X, eigenvals=E, conv=None, current_iteration=count, filename=filename,
         #                    choices=choices, precomputed_pca=precomputed_pca, u=None, G_i=None)
        X_prev = X
    print('converged: ' + str(count))
    return X, E, count