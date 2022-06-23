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


# utils
def autoregression_matrix(X, periods=1, fill_value=0):
    shifted_x = [pd.DataFrame(X).shift(periods=i, fill_value=fill_value).values for i in range(periods)]
    X_auto = np.hstack(tuple(shifted_x))
    return X_auto

def reference_test(X, window_size=2, step=1):
    T = []
    reference = []
    test = []
    for i in range(2*window_size-1, len(X), step):
        T.append(i)
        reference.append(X[i-2*window_size+1:i-window_size+1])
        test.append(X[i-window_size+1:i+1])
    return np.array(T), np.array(reference), np.array(test)

def KL_score_unsym(ref_ratios, test_ratios):
    score = np.mean(np.log(test_ratios))
    return score

def KL_score(ref_ratios, test_ratios):
    score = KL_score_unsym(ref_ratios, test_ratios) + KL_score_unsym(1./test_ratios, 1./ref_ratios)
    return score

def PE_score_unsym(ref_ratios, test_ratios, alpha=0.):
    score = (-0.5 *       alpha  * np.mean(test_ratios**2)) + \
            (-0.5 * (1. - alpha) * np.mean(ref_ratios**2))  + np.mean(test_ratios) - 0.5
    return score

def PE_score(ref_ratios, test_ratios, alpha=0.):
    score = PE_score_unsym(ref_ratios, test_ratios, alpha)# - PE_score_unsym(test_ratios, ref_ratios, alpha)
    return score

def KL(ref_preds, test_preds):
    return np.mean(np.log(test_preds + 10**-3)) - np.mean(np.log(1. - test_preds + 10**-3))

def KL_sym(ref_preds, test_preds):
    return np.mean(np.log(test_preds + 10**-3))     - np.mean(np.log(1. - test_preds + 10**-3)) + \
           np.mean(np.log(1. - ref_preds + 10**-3)) - np.mean(np.log(ref_preds + 10**-3))

def JSD(ref_preds, test_preds):
    return np.log(2) + 0.5 * np.mean(np.log(test_preds + 10**-3)) + 0.5 * np.mean(np.log(1. - ref_preds + 10**-3))

def PE(ref_preds, test_preds):
    scores = test_preds / (1. - test_preds + 10**-6) - 1.
    scores = np.clip(scores, 0, 1000)
    return np.mean(scores)

def PE_sym(ref_preds, test_preds):
    scores_1 = test_preds / (1. - test_preds + 10**-6) - 1.
    scores_1 = np.clip(scores_1, 0, 1000)
    scores_2 = (1. - ref_preds) / (ref_preds + 10**-6) - 1.
    scores_2 = np.clip(scores_2, 0, 1000)
    return np.mean(scores_1) + np.mean(scores_2)



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
        
        return np.mean(scores)


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

        inter = interpolate.interp1d(T_score-self.window_size, score, 
                                     kind='linear', fill_value=(0, 0), bounds_error=False)
        new_score = inter(T)
        peaks = argrelmax(new_score, order=self.window_size)[0]

        return new_score, peaks
        
    
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

        new_score, peaks = self.posprocessing(np.arange(len(X)), T_score, score)
        
        return new_score, peaks

    
    
    
# Classification
class ChangePointDetectionClassifier(ChangePointDetectionBase):
    
    def __init__(self, base_classifier=QuadraticDiscriminantAnalysis(), metric="KL_sym", 
                 periods=1, window_size=100, step=1, n_runs=1):
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.5, 
                                                            stratify=y, 
                                                            random_state=np.random.randint(0, 1000))
        
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
        
        return score
    


       
# NN RuLSIF
class ChangePointDetectionRuLSIF(ChangePointDetectionBase):
    
    def __init__(self, base_regressor, metric="PE", periods=1, window_size=100, step=1, n_runs=1):
        """
        Change point detection algorithm based on RuLSIF regressor.

        Parameters:
        -----------
        base_regressor: object
            Sklearn-like regressor with RuLSIF loss function.
        metric: string
            Name of the metric, that is used to measure the classifier quality and 
            considered as change point detection score. Default: "PE".
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
        self.base_regressor = base_regressor
        self.metric = metric


    def one_side_predict(self, X_train, X_test, y_train, y_test):
        """
        Fit a regressor on a pair of the train sample and make a prediction on the test.

        Parameters:
        -----------
        X_train: numpy.ndarray
            Matrix of train observations.
        X_test: numpy.ndarray
            Matrix of test observations.
        y_train: numpy.array
            Array of train labels.
        y_test: numpy.array
            Array of test labels.

        Retunrs:
        --------
        score: float
            Estimated change point detection score for the test.
        """

        reg = deepcopy(self.base_regressor)
        reg.fit(X_train, y_train)
        ratios = reg.predict(X_test)
        ref_ratios = ratios[y_test == 0]
        test_ratios = ratios[y_test == 1]
        if self.metric == "PE":
            score = 0.5 * np.mean(test_ratios) - 0.5 # PE_score_unsym(ref_ratios, test_ratios, classifier_1.alpha) - PE_score_unsym(test_ratios, ref_ratios, classifier_1.alpha)
        else:
            score = 0

        return score
        
        
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.5, 
                                                            stratify=y, 
                                                            random_state=np.random.randint(0, 1000))
        
        score_right = self.one_side_predict(X_train, X_test, y_train, y_test)
        score_left = self.one_side_predict(X_train, X_test, 1-y_train, 1-y_test)
        score = score_right + score_left
        
        return score

