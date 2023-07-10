from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from joblib import Parallel, delayed
from scipy import interpolate
from scipy.signal import argrelmax, savgol_filter

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from roerich.scores import maximum_mean_discrepancy, frechet_distance

from roerich.algorithms.models import GBDTRuLSIFRegressor, NNRuLSIFRegressor


# utils
def autoregression_matrix(X, periods=1, fill_value=0):
    shifted_x = [pd.DataFrame(X).shift(periods=i, fill_value=fill_value).values for i in range(periods)]
    X_auto = np.hstack(tuple(shifted_x))
    return X_auto


def reference_test(X, window_size=2, step=1):
    T = []
    reference = []
    test = []
    for i in range(2 * window_size - 1, len(X), step):
        T.append(i)
        reference.append(X[i - 2 * window_size + 1:i - window_size + 1])
        test.append(X[i - window_size + 1:i + 1])
    return np.array(T), np.array(reference), np.array(test)


def KL(ref_preds, test_preds):
    return np.mean(np.log(test_preds + 10 ** -3)) - np.mean(np.log(1. - test_preds + 10 ** -3))


def KL_sym(ref_preds, test_preds):
    return np.mean(np.log(test_preds + 10 ** -3)) - np.mean(np.log(1. - test_preds + 10 ** -3)) + \
        np.mean(np.log(1. - ref_preds + 10 ** -3)) - np.mean(np.log(ref_preds + 10 ** -3))


def JSD(ref_preds, test_preds):
    return np.log(2) + 0.5 * np.mean(np.log(test_preds + 10 ** -3)) + 0.5 * np.mean(np.log(1. - ref_preds + 10 ** -3))


def PE(ref_preds, test_preds):
    scores = test_preds / (1. - test_preds + 10 ** -6) - 1.
    scores = np.clip(scores, 0, 1000)
    return np.mean(scores)


def PE_sym(ref_preds, test_preds):
    scores_1 = test_preds / (1. - test_preds + 10 ** -6) - 1.
    scores_1 = np.clip(scores_1, 0, 1000)
    scores_2 = (1. - ref_preds) / (ref_preds + 10 ** -6) - 1.
    scores_2 = np.clip(scores_2, 0, 1000)
    return np.mean(scores_1) + np.mean(scores_2)


# Base
class ChangePointDetectionBase(metaclass=ABCMeta):

    def __init__(self, periods=1, window_size=100, step=1, n_runs=1):
        """
        Change point detection algorithm based on binary classification.

        Parameters:
        -----------
        periods: int, default=1
            Number of consecutive observations of a time series, considered as one input vector.
        The signal is considered as an autoregression process (AR) for classification. In the most cases periods=1
        will be a good choice.

        window_size: int, default=100
            Number of consecutive observations of a time series in test and reference
        windows. Recommendation: select the value so that there is only one change point within 2*window_size
        observations of the signal.

        step: int, default=1
            Algorithm estimates change point detection score for each <step> observation. step > 1 helps
        to speed up the algorithm.

        n_runs: int, default=1
            Number of times, the binary classifier runs on each pair of test and reference
        windows. Observations in the windows are divided randomly between train and validation sample every time.
        n_runs > 1 helps to reduce noise in the change point detection score.
        
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

        Returns:
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

    def postprocessing(self, T, T_score, score):
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

        inter = interpolate.interp1d(T_score - self.window_size, score,
                                     kind='linear', fill_value=(0, 0), bounds_error=False)
        score_new = inter(T)
        length = max(min(21, self.window_size//2), 5)
        score_smooth = savgol_filter(score_new, length, 3)
        peaks = argrelmax(score_smooth, order=self.window_size)[0]

        return score_smooth, peaks

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

        # data preprocessing
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)

        # change point score estimation
        iters = range(0, len(reference), self.step)
        score = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_score = np.array([T[i] for i in iters])

        # cpd score postprocessing
        new_score, peaks = self.postprocessing(np.arange(len(X)), T_score, score)

        return new_score, peaks


# Classification
class ChangePointDetectionClassifier(ChangePointDetectionBase):

    def __init__(self, base_classifier='mlp', metric="klsym",
                 periods=1, window_size=100, step=1, n_runs=1):
        """

        Change point detection algorithm based on binary classification [1]. It takes two sliding windows
        (reference and test) in a signal, and separate them using the classifier. The classification quality is
        considered as a change point detection score.

        [1] Mikhail Hushchyn and Andrey Ustyuzhanin. “Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation.” J. Comput. Sci. 53 (2021): 101385.

        Parameters:
        -----------
        base_classifier: {'logreg', 'qda', 'dt', 'rf', 'mlp', 'knn', 'nb'} or callable, default='mlp'
            Sklearn-like binary classifier to separate reference and test sliding windows in the signal.

            - 'logreg', Logistic Regression classifier,
              sklearn.linear_model.LogisticRegression()

            - 'qda', Quadratic Discriminant Analysis classifier,
              sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=0.01)

            - 'dt', Decision Tree classifier,
              sklearn.tree.DecisionTreeClassifier(min_samples_leaf=10, max_depth=6)

            - 'rf', Random Forest classifier,
              sklearn.ensemble.RandomForestClassifier((n_estimators=100, min_samples_leaf=10))

            - 'mlp', Multilayer Perceptron classifier,
              sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,100), solver="adam", activation="relu", learning_rate_init=0.1, max_iter=50, alpha=1.)

            - 'knn', K Neighbors classifier,
              sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)

            - 'nb', Gaussian Naive Bayes classifier,
              sklearn.naive_bayes.GaussianNB(var_smoothing=0.01)

            - Callable sklearn-like classifier,
              Example: base_classifier = LogisticRegression()

        metric: {'klsym', 'pesym', 'jsd', 'mmd', 'fd'} or callable, default='klsym'.
            {'KL', 'PE'} will be deprecated in future versions.
            Name of a score function, that is used to measure the classifier quality based on predictions
            for reference (p_ref) and test (p_test) windows. It is considered as change point detection score.

            - 'klsym' or 'KL_sym', symmetric Kullback-Leibler (KL) divergence,
            KL(p_test||p_ref) + KL(p_ref||p_test)

            - 'pesym' or 'PE_sym', symmetric Pearson (PE) divergence,
            PE(p_test||p_ref) + PE(p_ref||p_test)

            - 'jsd' or 'JSD', Jensen–Shannon divergence (JSD),
            JSD(p_test||p_ref)

            - 'mmd', the Maximum Mean Discrepancy (MMD),
            MMD(p_test, p_ref)

            - 'fd', the Frechet Distance (FD),
            FD(p_test, p_ref)

            - Callable function,
            Example: metric = roerich.scores.frechet_distance

        periods: int, default=1
            Number of consecutive observations of a time series, considered as one input vector.
        The signal is considered as an autoregression process (AR) for classification. In the most cases periods=1
        will be a good choice.

        window_size: int, default=100
            Number of consecutive observations of a time series in test and reference
        windows. Recommendation: select the value so that there is only one change point within 2*window_size
        observations of the signal.

        step: int, default=1
            Algorithm estimates change point detection score for each <step> observation. step > 1 helps
        to speed up the algorithm.

        n_runs: int, default=1
            Number of times, the binary classifier runs on each pair of test and reference
        windows. Observations in the windows are divided randomly between train and validation sample every time.
        n_runs > 1 helps to reduce noise in the change point detection score.

        """

        super().__init__(periods, window_size, step, n_runs)

        nsam = min(10, window_size // 4 + 1)
        if base_classifier == 'qda':
            self.base_classifier = QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=0.01)
        elif base_classifier == 'logreg':
            self.base_classifier = LogisticRegression()
        elif base_classifier == 'dt':
            self.base_classifier = DecisionTreeClassifier(min_samples_leaf=nsam, max_depth=6)
        elif base_classifier == 'rf':
            self.base_classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=nsam)
        elif base_classifier == 'mlp':
            self.base_classifier = MLPClassifier(hidden_layer_sizes=(100, 100), solver="adam", activation="relu",
                                                 learning_rate_init=0.1, max_iter=50, alpha=1.)
        elif base_classifier == 'knn':
            self.base_classifier = KNeighborsClassifier(n_neighbors=nsam)
        elif base_classifier == 'nb':
            self.base_classifier = GaussianNB(var_smoothing=0.01)
        else:
            self.base_classifier = base_classifier

        if metric == "KL" or metric == "kl":
            self.metric = KL
        elif metric == "KL_sym" or metric == "klsym":
            self.metric = KL_sym
        elif metric == "JSD" or metric == "jsd":
            self.metric = JSD
        elif metric == "PE" or metric == "pe":
            self.metric = PE
        elif metric == "PE_sym" or metric == "pesym":
            self.metric = PE_sym
        elif metric == "mmd":
            self.metric = maximum_mean_discrepancy
        elif metric == "fd":
            self.metric = frechet_distance
        else:
            self.metric = metric

    @ignore_warnings(category=ConvergenceWarning)
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

        X = np.vstack((X_ref, X_test))
        y = np.hstack((np.zeros(len(X_ref)), np.ones(len(X_test))))

        ss = StandardScaler()
        X = ss.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.5,
                                                            stratify=y,
                                                            random_state=np.random.randint(0, 1000))

        classifier = deepcopy(self.base_classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        score = self.metric(y_pred[y_test == 0], y_pred[y_test == 1])

        return score


# NN RuLSIF
class ChangePointDetectionRuLSIF(ChangePointDetectionBase):

    def __init__(self, base_regressor='mlp', metric="pesym", periods=1, window_size=100, step=1, n_runs=1):
        """

        Change point detection algorithm based on RuLSIF regressor [1]. It takes two sliding windows
        (reference and test) in a signal, and directly estimates probability density ratio. The ratios are used to 
        calculate the change point detection score.

        [1] Mikhail Hushchyn and Andrey Ustyuzhanin. “Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation.” J. Comput. Sci. 53 (2021): 101385.

        Parameters:
        -----------
        base_regressor: {'mlp', 'gbdt'} or callable, default='mlp'
            Sklearn-like regressor with the RuLSIF loss function to estimate the density ratio.

            - 'mlp', Multilayer Perceptron regressor with RuLSIF loss function,
              roerich.algorithms.models.NNRuLSIFRegressor(n_hidden=100, n_epochs=50, batch_size=2*window_size, lr=0.1, l2=0.01, alpha=0.05)
            
            - 'gbdt', Gradient Boosting over Decision Trees regressor with RuLSIF loss function,
              roerich.algorithms.models.GBDTRuLSIFRegressor(n_estimators=50, learning_rate=0.2, max_depth=3, alpha=0.05, min_samples_leaf=nsam)
              
            - Callable object,
            Example: base_regressor = roerich.algorithms.models.NNRuLSIFRegressor()

        metric: {'pesym'}, default='pesym'.
            Name of a score function, that is used to calculate the change point detection score based on predictions
            for reference (p_ref) and test (p_test) windows.

            - 'pesym' or 'PE_sym', Pearson (PE) divergence,
            PE(p_test||p_ref) + PE(p_ref||p_test)

        periods: int, default=1
            Number of consecutive observations of a time series, considered as one input vector.
        The signal is considered as an autoregression process (AR) for regression. In the most cases periods=1
        will be a good choice.

        window_size: int, default=100
            Number of consecutive observations of a time series in test and reference
        windows. Recommendation: select the value so that there is only one change point within 2*window_size
        observations of the signal.

        step: int, default=1
            Algorithm estimates change point detection score for each <step> observation. step > 1 helps
        to speed up the algorithm.

        n_runs: int, default=1
            Number of times, the regressor runs on each pair of test and reference
        windows. Observations in the windows are divided randomly between train and validation sample every time.
        n_runs > 1 helps to reduce noise in the change point detection score.

        """

        super().__init__(periods, window_size, step, n_runs)

        nsam = min(10, window_size // 4 + 1)
        if base_regressor == 'gbdt':
            self.base_regressor = GBDTRuLSIFRegressor(n_estimators=50, learning_rate=0.2,
                                                      max_depth=3, alpha=0.05, min_samples_leaf=nsam)
        elif base_regressor == 'mlp':
            self.base_regressor = NNRuLSIFRegressor(n_hidden=100, n_epochs=50,
                                                    batch_size=2*window_size, lr=0.1, l2=0.01, alpha=0.05)
        else:
            self.base_regressor = base_regressor

        self.metric = metric

    def _one_side_predict(self, X_train, X_test, y_train, y_test):
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
        ratios = reg.predict_proba_ratio(X_test)

        if self.metric == "PE" or self.metric == "PE_sym" or self.metric == "pesym":
            score = 0.5 * np.mean(ratios[y_test == 1]) - 0.5
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

        ss = StandardScaler()
        X = ss.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.5,
                                                            stratify=y,
                                                            random_state=np.random.randint(0, 1000))

        score_right = self._one_side_predict(X_train, X_test, y_train, y_test)
        score_left = self._one_side_predict(X_test, X_train, 1 - y_test, 1 - y_train)
        score = score_right + score_left

        return score
