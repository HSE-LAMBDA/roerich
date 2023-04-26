from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from copy import deepcopy
from joblib import Parallel, delayed
from scipy import interpolate
from scipy.signal import argrelmax

from .cpdc import ChangePointDetectionBase

from sklearn.model_selection import StratifiedKFold, KFold


# Classification
class ChangePointDetectionClassifierCV(ChangePointDetectionBase):
    
    def __init__(self, base_classifier=QuadraticDiscriminantAnalysis(), metric="KL_sym", 
                 periods=1, window_size=100, step=1, n_runs=1, n_splits=5):

        """
        Change point detection algorithm based on binary classififcation with cross validation.

        Parameters:
        -----------
        base_classifier: object
            Sklearn-like binary classifier.
        metric: string
            Name of the metric, that is used to measure the classifier quality and 
            considered as change point detection score. Default: "KL_sym".
        periods: int
            Number of consecutive observations of a time series, considered as one input vector.
        window_size: int
            Number of consecutive observations of a time series in test and reference windows.
        step: int
            Algorithm estimates change point detection score for each <step> observation.
        n_runs: int
            Number of times, the binary classifier runs on each pair of test and reference windows.
        n_splits: int
            Number of splits for cross validation.
        """

        super().__init__(periods, window_size, step, n_runs)
        self.base_classifier = base_classifier
        self.metric = metric
        self.n_splits = n_splits
        

    def densratio(self, y, alpha=10**-3):
        w = (y + alpha) / (1 - y + alpha)
        return w
        
        
    def reference_test_predict(self, X_ref, X_test):
        """
        Estimate change point detection score for a pair of test and reference windows.

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

        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=np.random.randint(0, 1000))
        scores = []

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test, y_train, y_test = (X[train_index], 
                                                X[test_index], 
                                                y[train_index], 
                                                y[test_index])
            
            classifier = deepcopy(self.base_classifier)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict_proba(X_test)[:, 1]
            
            ref_preds = y_pred[y_test == 0]
            test_preds = y_pred[y_test == 1]
            ratios = self.densratio(y_pred)
            ref_ratios = ratios[y_test == 0]
            test_ratios = ratios[y_test == 1]
            
            if self.metric == "KL":
                score = KL(ref_preds, test_preds)
            elif self.metric == "KL_sym":
                score = KL_sym(ref_preds, test_preds)
            elif self.metric == "JSD":
                score = JSD(ref_preds, test_preds)
            elif self.metric == "PE":
                score = PE(ref_preds, test_preds)
            elif self.metric == "PE_sym":
                score = PE_sym(ref_preds, test_preds)
            elif self.metric == "ROCAUC":
                score = 2 * (roc_auc_score(y_test, y_pred) - 0.5)
            else:
                score = 0

            scores.append(score)

        return np.mean(scores)