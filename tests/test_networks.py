import numpy as np
import pytest
from inspect import getmembers, isfunction, isclass
from roerich import networks
import torch

models = [f[1] for f in getmembers(networks, isclass)]

@pytest.mark.parametrize("model", models)
def test_base(model):
    # generate samples
    n = 10
    n_inputs = 2
    X = torch.rand(size=(n, n_inputs))

    # make predictions
    reg = model(n_inputs=n_inputs)
    y_pred = reg(X)
    assert y_pred.size() == (n, 1)
