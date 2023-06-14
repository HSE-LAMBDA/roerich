import numpy as np
import pytest
from inspect import getmembers, isfunction, isclass
from roerich import change_point

models = [f[1] for f in getmembers(change_point, isclass)]

@pytest.mark.parametrize("model", models)
def test_base(model):
    # generate time series
    n = 400
    X1 = np.random.normal(0, 1, size=(n, 1))
    X2 = np.random.normal(2, 1, size=(n, 1))
    X = np.concatenate((X1, X2), axis=0)
    cps_true = [n]

    # detection
    cp = model()
    score, cps_pred = cp.predict(X)
