import numpy as np
import pytest
from inspect import getmembers, isfunction
from roerich import scores

funcs = [f[1] for f in getmembers(scores, isfunction)]


@pytest.mark.parametrize("cost", funcs)
def test_same_in_size_2d(cost):
    N = 100
    X_real = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], N)
    X_fake = np.random.multivariate_normal([0, 0.], [[1, 0.7], [0.7, 1]], N)
    score = cost(X_real, X_fake)

@pytest.mark.parametrize("cost", funcs)
def test_same_in_size_1d(cost):
    N = 100
    X_real = np.random.normal(0, 1, N).reshape(-1, 1)
    X_fake = np.random.normal(1, 1, N).reshape(-1, 1)
    score = cost(X_real, X_fake)

@pytest.mark.parametrize("cost", funcs)
def test_different_in_size_2d(cost):
    X_real = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 100)
    X_fake = np.random.multivariate_normal([0, 0.], [[1, 0.7], [0.7, 1]], 153)
    score = cost(X_real, X_fake)

    X_real = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 342)
    X_fake = np.random.multivariate_normal([0, 0.], [[1, 0.7], [0.7, 1]], 100)
    score = cost(X_real, X_fake)