import numpy as np
from .cpdc import autoregression_matrix
from scipy.signal import argrelmax, savgol_filter
from scipy import interpolate

import torch
import torch.nn as nn
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


class OnlineNNRuLSIF(object):

    def __init__(self, net='auto', scaler='auto', alpha=0.1, metric="pesym", window_size=1, lag_size=100, periods=1,
                 step=1, n_epochs=1, lr=0.1, lam=0., optimizer="RMSprop"):
        """

        Change point detection algorithm using RuLSIF regression based on online neural network [1].
        It takes two small (1-10) sliding windows (reference and test) in a signal, and directly estimates probability density ratio.
        The ratios are used to calculate the change point detection score.

        [1] Mikhail Hushchyn, Kenenbek Arzymatov and Denis Derkach. “Online Neural Networks for Change-Point Detection.” ArXiv abs/2010.01388 (2020)

        Parameters:
        -----------
        net: {'auto'} or callable, default='auto'
            PyTorch neural network for binary classification to separate reference and test sliding windows in the signal.

            - 'auto', two-layers neural network with 100 neurons in each layer, and ReLU activation.
              roerich.algorithms.onnr.BaseNN()

            - Callable, pytorch neural network for regression,
              Example: base_classifier = roerich.algorithms.onnr.BaseNN

        scaler: {'auto'}, default='auto'
            Ignored. It will be deprecated in future versions.

        metric: {'pesym'} or callable, default='pesym'.
            Name of a score function, that is used to measure the classifier quality based on predictions
            for reference (p_ref) and test (p_test) windows. It is considered as change point detection score.

            - 'pesym' or 'PE_sym', symmetric Pearson (PE) divergence,
            PE(p_test||p_ref) + PE(p_ref||p_test)

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

        self.net1 = None
        self.net2 = None
        self.alpha = alpha
        self.metric = metric
        self.window_size = window_size
        self.periods = periods
        self.lag_size = lag_size
        self.step = step
        self.n_epochs = n_epochs
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer

    def _init_networks(self, n_inputs):

        # init network
        self.net1 = self.base_net(n_inputs)
        self.net2 = self.base_net(n_inputs)

        # init optimizer
        if self.optimizer == "Adam":
            self.opt1 = torch.optim.Adam(self.net1.parameters(), lr=self.lr, weight_decay=self.lam)
            self.opt2 = torch.optim.Adam(self.net2.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "SGD":
            self.opt1 = torch.optim.SGD(self.net1.parameters(), lr=self.lr, weight_decay=self.lam)
            self.opt2 = torch.optim.SGD(self.net2.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "RMSprop":
            self.opt1 = torch.optim.RMSprop(self.net1.parameters(), lr=self.lr, weight_decay=self.lam)
            self.opt2 = torch.optim.RMSprop(self.net2.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "ASGD":
            self.opt1 = optim.ASGD(self.net1.parameters(), lr=self.lr, lambd=0.0, alpha=0.75, t0=0.0,
                                   weight_decay=self.lam)
            self.opt2 = optim.ASGD(self.net2.parameters(), lr=self.lr, lambd=0.0, alpha=0.75, t0=0.0,
                                   weight_decay=self.lam)
        else:
            self.opt1 = torch.optim.Adam(self.net1.parameters(), lr=self.lr, weight_decay=self.lam)
            self.opt2 = torch.optim.Adam(self.net2.parameters(), lr=self.lr, weight_decay=self.lam)

    def _rulsif_loss(self, y_pred_ref, y_pred_test, alpha):
        loss = 0.5 * (1 - alpha) * (y_pred_ref ** 2).mean() + \
               0.5 * alpha * (y_pred_test ** 2).mean() - (y_pred_test).mean()
        return loss

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
        self._init_networks(X_auto.shape[1])

        # cpd score
        score = [self._reference_test_predict(reference[i], test[i]) for i in range(len(reference))]
        T_score = np.array([T[i] for i in range(len(reference))])

        # cpd score postprocessing
        new_score, peaks = self._postprocessing(np.arange(len(X)), T_score, score)

        return new_score, peaks

    def _reference_test_predict(self, X_ref, X_test):

        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        n_last = min(self.window_size, self.step)
        self.net1.train(False)
        test_preds = self.net1(X[y == 1][-n_last:]).detach().numpy()

        self.net2.train(False)
        ref_preds = self.net2(X[y == 0][-n_last:]).detach().numpy()

        self.net1.train(True)
        self.net2.train(True)
        for epoch in range(self.n_epochs):  # loop over the dataset multiple times

            # forward + backward + optimize
            y_pred_batch = self.net1(X).squeeze()
            y_pred_batch_ref = y_pred_batch[y == 0]
            y_pred_batch_test = y_pred_batch[y == 1]
            loss = self._rulsif_loss(y_pred_batch_ref, y_pred_batch_test, self.alpha)
            self.opt1.zero_grad()
            loss.backward()
            self.opt1.step()

            # forward + backward + optimize
            y_pred_batch = self.net2(X).squeeze()
            y_pred_batch_ref = y_pred_batch[y == 1]
            y_pred_batch_test = y_pred_batch[y == 0]
            loss = self._rulsif_loss(y_pred_batch_ref, y_pred_batch_test, self.alpha)
            self.opt2.zero_grad()
            loss.backward()
            self.opt2.step()

        score = (0.5 * np.mean(test_preds) - 0.5) + (0.5 * np.mean(ref_preds) - 0.5)

        return score

    def _reference_test(self, X):
        N = self.lag_size
        ws = self.window_size
        T = []
        reference = []
        test = []
        for i in range(2 * ws + N - 1, len(X), self.step):
            T.append(i)
            reference.append(X[i - 2 * ws - N + 1:i - ws - N + 1])
            test.append(X[i - ws + 1:i + 1])
        return np.array(T), np.array(reference), np.array(test)
