import numpy as np
from sklearn.metrics import pairwise_distances


def energy_distance(x, y):
    n = x.shape[0]
    e = 2*pairwise_distances(x, y, metric='euclidean')\
        - pairwise_distances(y, metric='euclidean') \
        - pairwise_distances(x, metric='euclidean')
    return np.sum(e) / n ** 2
