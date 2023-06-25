import numpy as np
import pytest
from inspect import getmembers, isfunction, isclass
from roerich import change_point
from sklearn.linear_model import LogisticRegression
from roerich.costs import frechet_distance

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

cpdc = [change_point.ChangePointDetectionClassifier]
@pytest.mark.parametrize("model", cpdc)
def test_base_classifier(model):
    # generate time series
    n = 200
    X1 = np.random.normal(0, 1, size=(n, 1))
    X2 = np.random.normal(2, 1, size=(n, 1))
    X = np.concatenate((X1, X2), axis=0)
    cps_true = [n]

    # test clf
    lr = LogisticRegression()
    for clf in ['logreg', 'qda', 'dt', 'rf', 'mlp', 'knn', 'nb', lr]:
        cp = model(base_classifier=clf, window_size=50)
        score, cps_pred = cp.predict(X)

@pytest.mark.parametrize("model", cpdc)
def test_metric(model):
    # generate time series
    n = 200
    X1 = np.random.normal(0, 1, size=(n, 1))
    X2 = np.random.normal(2, 1, size=(n, 1))
    X = np.concatenate((X1, X2), axis=0)
    cps_true = [n]

    # test clf
    fd = frechet_distance
    for m in ['klsym', 'pesym', 'jsd', 'mmd', 'fd', fd]:
        cp = model(base_classifier='logreg', window_size=50, metric=m)
        score, cps_pred = cp.predict(X)
