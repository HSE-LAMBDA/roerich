from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from copy import deepcopy
from joblib import Parallel, delayed
from scipy import interpolate
from scipy.signal import argrelmax

from .cpdc import ChangePointDetectionBase
from roerich.scores.fd import frechet_distance
from roerich.scores.mmd import maximum_mean_discrepancy


class ScoreCalculator(ChangePointDetectionBase):

    def __init__(self, metric=None, func=None, periods=1, window_size=100, step=1, n_runs=1):
        super().__init__(periods=periods, window_size=window_size, step=step, n_runs=n_runs)
        self.metric = metric
        self.func = func

    def reference_test_predict(self, X_ref, X_test):

        if self.metric == "EnergyDist":
            n = X_ref.shape[0]
            E = 2*pairwise_distances(X_ref, X_test, metric='euclidean') - pairwise_distances(X_test, metric='euclidean') - pairwise_distances(X_ref, metric='euclidean')
            return np.sum(E) / n ** 2
        elif self.metric == "FrechetDist":
            return frechet_distance(X_ref, X_test)
        elif self.metric == "MaxMeanDisc":
            return maximum_mean_discrepancy(X_ref, X_test)
        elif self.func is not None:
            return self.func(X_ref, X_test)
        else:
            raise ValueError("metric should be one of: EnergyDist, FrechetDist, MaxMeanDisc; or a function should be "
                             "passed")
