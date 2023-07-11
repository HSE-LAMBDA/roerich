import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import argrelmax, savgol_filter

import torch
import torch.nn as nn
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# utils
def autoregression_matrix(X, periods=1, fill_value=0):
    shifted_x = [pd.DataFrame(X).shift(periods=i, fill_value=fill_value).values for i in range(periods)]
    X_auto = np.hstack(tuple(shifted_x))
    return X_auto


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


def SMA(scores, N):
    new_scores = []
    for i in range(0, len(scores)):
        s = i - N if i - N >= 0 else 0
        new_scores.append(np.mean(scores[s:i + 1], axis=0))
    return np.array(new_scores)


class BaseNN(nn.Module):
    def __init__(self, n_inputs=1):
        super(BaseNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 100)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class OnlineNNClassifier(object):

    def __init__(self, net='auto', scaler='auto', metric="klsym", window_size=1, lag_size=100, periods=1, step=1,
                 n_epochs=1, lr=0.1, lam=0., optimizer="RMSprop"):
        """

        Change point detection algorithm using binary classification based on online neural network [1].
        It takes two small (1-10) sliding windows (reference and test) in a signal, and separate them using the classifier,
        that is trained online. The classification quality is considered as a change point detection score.

        [1] Mikhail Hushchyn, Kenenbek Arzymatov and Denis Derkach. “Online Neural Networks for Change-Point Detection.” ArXiv abs/2010.01388 (2020)

        Parameters:
        -----------
        net: {'auto'} or callable, default='auto'
            PyTorch neural network for binary classification to separate reference and test sliding windows in the signal.

            - 'auto', two-layers neural network with 100 neurons in each layer, and ReLU activation.
              roerich.algorithms.onnc.BaseNN()

            - Callable, pytorch neural network for classification,
              Example: base_classifier = roerich.algorithms.onnc.BaseNN

        scaler: {'auto'}, default='auto'
            Ignored. It will be deprecated in future versions.

        metric: {'klsym', 'pesym', 'jsd'} or callable, default='klsym'.
            {'KL', 'PE'} will be deprecated in future versions.
            Name of a score function, that is used to measure the classifier quality based on predictions
            for reference (p_ref) and test (p_test) windows. It is considered as change point detection score.

            - 'klsym' or 'KL_sym', symmetric Kullback-Leibler (KL) divergence,
            KL(p_test||p_ref) + KL(p_ref||p_test)

            - 'pesym' or 'PE_sym', symmetric Pearson (PE) divergence,
            PE(p_test||p_ref) + PE(p_ref||p_test)

            - 'jsd' or 'JSD', Jensen–Shannon divergence (JSD),
            JSD(p_test||p_ref)

            - Callable function,
            Example: metric = roerich.scores.frechet_distance

        window_size: int, default=1
            Number of consecutive observations of a time series in test and reference
        windows.

        lag_size: int, default=100
            Number of observations between the reference and test windows. Recommendation: select the value so that
        there is only one change point within 2*lag_size observations of the signal.

        periods: int, default=1
            Number of consecutive observations of a time series, considered as one input vector.
        The signal is considered as an autoregression process (AR) for classification. In the most cases periods=1
        will be a good choice.

        step: int, default=1
            Algorithm estimates change point detection score for each <step> observation. step > 1 helps
        to speed up the algorithm.

        n_epochs: int, default=1
            Number of training epochs per each pair of windows.

        lr: float, default=0.1
            Learning rate.

        lam: float, default=0
            L2 regularization.

        optimizer: {"Adam", "SGD", "RMSprop", "ASGD"}, default="RMSprop"
            Optimizer used to fit the online neural network.

        """

        if net == 'auto':
            self.base_net = BaseNN
        else:
            self.base_net = net

        self.net = None
        self.metric = metric
        self.window_size = window_size
        self.periods = periods
        self.lag_size = lag_size
        self.step = step
        self.n_epochs = n_epochs
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer

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
        else:
            self.metric = KL_sym

    def _init_network(self, n_inputs):

        # init network
        self.net = self.base_net(n_inputs)

        # init optimizer
        if self.optimizer == "Adam":
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "SGD":
            self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "RMSprop":
            self.opt = torch.optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "ASGD":
            self.opt = optim.ASGD(self.net.parameters(), lr=self.lr, lambd=0.0, alpha=0.75, t0=0.0, weight_decay=self.lam)
        else:
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.lam)

        # loss function
        self.criterion = nn.BCELoss()

    def _reference_test(self, X):
        lag = self.lag_size
        ws = self.window_size
        T = []
        reference = []
        test = []
        for i in range(2 * ws + lag - 1, len(X), self.step):
            T.append(i)
            reference.append(X[i - 2 * ws - lag + 1:i - ws - lag + 1])
            test.append(X[i - ws + 1:i + 1])
        return np.array(T), np.array(reference), np.array(test)

    def _reference_test_predict(self, X_ref, X_test):
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

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        # evaluate
        self.net.train(False)
        n_last = min(self.window_size, self.step)
        ref_preds = self.net(X[y == 0][-n_last:]).detach().numpy()
        test_preds = self.net(X[y == 1][-n_last:]).detach().numpy()

        # change point detection score
        score = self.metric(ref_preds, test_preds)

        # update network
        self.net.train(True)
        for epoch in range(self.n_epochs):
            outputs = self.net(X)
            loss = self.criterion(outputs.squeeze(), y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return score

    def _postprocessing(self, T, T_score, score):
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

        shift = self.lag_size + self.window_size
        inter = interpolate.interp1d(T_score - shift, score,
                                     kind='linear', fill_value=(0, 0), bounds_error=False)
        score = inter(T)

        score = SMA(score, shift)

        length = max(min(21, shift // 2), 5)
        score = savgol_filter(score, length, 3)
        peaks = argrelmax(score, order=shift)[0]

        return score, peaks

    def predict(self, X, distance=5, height=None, smooth=False):
        """
        Estimate change point detection score for a time series.

        Parameters:
        -----------
        X: numpy.ndarray
            Time series observation.

        distance: int, default=5
            Ignored. It will be deprecated in future versions.

        height: float or None, default=None
            Ignored. It will be deprecated in future versions.

        smooth: {True, False}, default=False
            Ignored. It will be deprecated in future versions.

        Retunrs:
        --------
        T_score: numpy.array
            Array of timestamps.
        scores: numpy.array
            Estimated change point detection score.
        """

        # data preparation
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = self._reference_test(X_auto)

        # init network
        self._init_network(X_auto.shape[1])

        # cpd score
        score = [self._reference_test_predict(reference[i], test[i]) for i in range(len(reference))]
        T_score = np.array([T[i] for i in range(len(reference))])

        # cpd score postprocessing
        new_score, peaks = self._postprocessing(np.arange(len(X)), T_score, score)

        return new_score, peaks
