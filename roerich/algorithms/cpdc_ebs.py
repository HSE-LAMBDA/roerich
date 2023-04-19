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

from .cpdc import *

from sklearn.model_selection import StratifiedKFold, KFold

# Base
class ChangePointDetectionBase(metaclass=ABCMeta):
    
    def __init__(self, periods=1, window_size=100, step=1, n_runs=1):
        """
        Change point detection algorithm based on binary classififcation.

        Parameters:
        -----------
        periods: int
            Number of consecutive observations of a time series, considered as one input vector.
        window_size: int
            Number of consecutive observations of a time series in test and reference windows.
        step: int
            Algorithm estimates change point detection score for each <step> observation.
        n_runs: int
            Number of times, the binary classifier runs on each pair of test and reference windows.
        
        """

        self.periods = periods
        self.window_size = window_size
        self.step = step
        self.n_runs = n_runs
        
        
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

        score = 0
        
        return score
    
    
    def reference_test_predict_n_times(self, X_ref, X_test):
        """
        Estimate change point detection score several times for  a pair of test and reference windows.

        Parameters:
        -----------
        X_ref: numpy.ndarray
            Matrix of reference observations.
        X_test: numpy.ndarray
            Matrix of test observations.

        Returns:
        --------
        score: float
            Estimated average change point detection score for a pair of window.
        """

        scores = []
        for i in range(self.n_runs):
            ascore = self.reference_test_predict(X_ref, X_test)
            scores.append(ascore)
        
        return np.mean(scores, axis=0)


    def posprocessing(self, T, T_score, score):
        """
        Interpolates and shifts a change point detection score, estimates peak positions.
        
        Parameters:
        -----------
        T: numpy.array
            A broader time-step interval
        T_score: numpy.array
            A time intervals of CPD scores
        score: numpy.array
            A CPD scores

        Returns:
        --------
        new_score: numpy.array
            Interpolated and shifted CPD scores.
        peaks: numpy.array
            Positions of peaks in the CPD scores.
        """

        new_scores = []
        peaks = []
        
        for j in range(self.n_samples):
            inter = interpolate.interp1d(T_score-self.window_size, score[:, j], 
                                         kind='linear', fill_value=(0, 0), bounds_error=False)
            new_score = inter(T)
            peak = argrelmax(new_score, order=self.window_size)[0]

            new_scores.append(new_score)
            peaks.append(peak)

        return new_scores, peaks
        
    
    def predict(self, X):
        """
        Estimate change point detection score for a time series.

        Parameters:
        -----------
        X: numpy.ndarray
            Time series observation.

        Retunrs:
        --------
        T_score: numpy.array
            Array of timestamps.
        scores: numpy.array
            Estimated change point detection score.
        """
        
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)
        
        scores = []
        T_score = []
        iters = range(0, len(reference), self.step)
        score = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_score = np.array([T[i] for i in iters])

        new_score, peaks = self.posprocessing(np.arange(len(X)), T_score, np.array(score))
        
        return new_score, peaks


# Classification
class ChangePointDetectionClassifierEBS(ChangePointDetectionBase):
    
    def __init__(self, base_classifier=QuadraticDiscriminantAnalysis(), metric="KL_sym", 
                 periods=1, window_size=100, step=1, n_runs=1, n_samples=100):

        """
        Change point detection algorithm based on binary classififcation.

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
        
        """

        super().__init__(periods, window_size, step, n_runs)
        self.base_classifier = base_classifier
        self.metric = metric
        self.n_samples = n_samples
        

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

        scores = []

        for j in range(self.n_samples):
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=0.5, 
                                                                stratify=y, 
                                                                random_state=np.random.randint(0, 1000))
            
            # Bootstrap
            ref = X_train[y_train == 0]
            test = X_train[y_train == 1]

            X_ref_bs = ref[np.random.choice(len(ref), size=len(ref), replace=True)]
            X_test_bs = test[np.random.choice(len(test), size=len(test), replace=True)]

            X_train = np.concatenate([X_ref_bs, X_test_bs])

            y_ref_bs = np.zeros(len(X_ref_bs))
            y_test_bs = np.ones(len(X_test_bs))

            y_train = np.concatenate([y_ref_bs, y_test_bs])

            shuffle_inds = np.random.choice(len(y_train), size=len(y_train), replace=False)

            X_train = X_train[shuffle_inds]
            y_train = y_train[shuffle_inds]

            # Classifier
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

        return scores