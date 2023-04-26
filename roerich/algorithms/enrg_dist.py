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

from .cpdc import ChangePointDetectionBase, KL, KL_sym, JSD, PE, PE_sym

# EnergyDistance
class EnergyDistanceCalculator(ChangePointDetectionBase):


    def reference_test_predict(self, X_ref, X_test):
        """
        Estimate change point detection score for a pair of test and reference windows by energy
        distance metric.

        Parameters:
        -----------
        X_ref: numpy.ndarray
            Matrix of reference observations.
        X_test: numpy.ndarray
            Matrix of test observations.

        Retunrs:
        --------
        score: float
            Estimated change point detection score for a pair of window.
        """

        n = X_ref.shape[0]
        E = 2*pairwise_distances(X_ref, X_test, metric='euclidean') - pairwise_distances(X_test, metric='euclidean') - pairwise_distances(X_ref, metric='euclidean')
        return np.sum(E)/**2