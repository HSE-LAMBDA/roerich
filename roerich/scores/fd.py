import numpy as np
from scipy.linalg import sqrtm


def frechet_distance(X, Y):
    '''
    Calculates the Frechet Distance between X and Y samples.

    Parameters:
    -----------
    X: numpy.ndarray of shape [n_samples, n_features]
        The first sample.
    Y: numpy.ndarray of shape [n_samples, n_features]
        The second sample.

    Return:
    -------
    distance: float
        The estimated Frechet Distance.
    '''

    # 1d array to nd one
    X = np.reshape(X, (len(X), -1))
    Y = np.reshape(Y, (len(Y), -1))

    X_mean, X_cov = X.mean(axis=0), np.cov(X, rowvar=False)
    Y_mean, Y_cov = Y.mean(axis=0), np.cov(Y, rowvar=False)

    if X.shape[1] == 1:
        X_cov = np.array([[X_cov]])
    if Y.shape[1] == 1:
        Y_cov = np.array([[Y_cov]])

    diff = np.sum((X_mean - Y_mean)**2.0)
    covmean, _ = sqrtm(X_cov.dot(Y_cov), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real

    distance = diff + np.trace(X_cov) + np.trace(Y_cov) - 2 * np.trace(covmean)

    return distance
