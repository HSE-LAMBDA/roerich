import numpy as np
from sklearn import metrics


def maximum_mean_discrepancy(X, Y):
    '''
    [Computationally expensive. Recommended for samples size < 5000]
    Calculates the Maximum Mean Discrepancy between X and Y samples.

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

    agg_matrix = np.concatenate((X, Y), axis=0)
    distances = metrics.pairwise_distances(agg_matrix)
    median_distance = np.median(distances) + 10**-10
    gamma = 1.0 / (2 * median_distance**2)

    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()

    return mmd
