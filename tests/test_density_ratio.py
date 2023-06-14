import numpy as np
import pytest
from scipy.stats import norm, uniform
from inspect import getmembers, isfunction, isclass
from roerich import density_ratio

models = [f[1] for f in getmembers(density_ratio, isclass)]

@pytest.mark.parametrize("model", models)
def test_base(model):
    # generate samples
    N = 1000
    p0 = uniform(-5, 10)
    p1 = norm(0, 1)
    X = np.concatenate((p0.rvs((N, 1)), p1.rvs((N, 1))))
    y = np.array([0] * N + [1] * N)

    # true density ratio
    true_ratio = p1.pdf(X) / p0.pdf(X)

    # direct density ratio estimation
    reg = model()
    reg.fit(X, y)
    pred_ratio = reg.predict_proba_ratio(X)
