import numpy as np
from sklearn.metrics import pairwise_distances


def energy_distance(x, y):
    x = np.reshape(x, (len(x), -1))
    y = np.reshape(y, (len(y), -1))
    n = x.shape[0]
    a = np.mean(pairwise_distances(x, y, metric='euclidean'))
    b = np.mean(pairwise_distances(y, metric='euclidean'))
    c = np.mean(pairwise_distances(x, metric='euclidean'))
    return 2 * a - b - c
