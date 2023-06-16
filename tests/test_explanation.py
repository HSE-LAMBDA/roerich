import numpy as np
import pytest
from roerich import explanation
from roerich import change_point

def test_matrix_importance():
    # generate time series
    n = 400
    X1 = np.random.normal([0, 0, 0], 1, size=(n, 3))
    X2 = np.random.normal([0, 1, 2], 1, size=(n, 3))
    X = np.concatenate((X1, X2), axis=0)
    cps_true = [n]

    # base detection algo
    cp = change_point.EnergyDistanceCalculator()

    # importance
    mm = explanation.MatrixImportance(cp)
    score, cps_pred = mm.predict(X)
    mat_u = mm.predict_union(X)
    mat_e = mm.predict_exclude(X)

    assert score.shape == (2 * n,)
    assert mat_u.shape == (2*n, 3, 3)
    assert mat_e.shape == (2 * n, 3, 3)
