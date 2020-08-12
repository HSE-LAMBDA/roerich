from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from scipy import interpolate
from scipy.signal import find_peaks_cwt

from .net import MyNN, MyNNRegressor
from .metrics import autoregression_matrix
from .metrics import KL_sym, KL, JSD, PE, PE_sym, Wasserstein
from .scaler import SmaScalerCache
from .helper import SMA


class ChangePointDetection(metaclass=ABCMeta):
    def __init__(self, scaler="default", metric="KL", window_size=1, periods=10, lag_size=0,
                 step=1, n_epochs=100, lr=0.01, lam=0, optimizer="Adam", debug=0):
        
        self.scaler = SmaScalerCache(window_size + lag_size) if scaler == "default" else scaler
        
        self.metric = metric
        self.window_size = window_size
        self.periods = periods
        self.lag_size = lag_size
        self.step = step
        self.n_epochs = n_epochs
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer
        self.debug = debug
        
        self._time_shift = lag_size + window_size
        self.avg_window = lag_size + window_size
        self.peak_widths = 0.25 * (lag_size + window_size)
        
        self.optimizers = defaultdict(lambda: torch.optim.Adam)
        
        self.optimizers["Adam"] = torch.optim.Adam
        self.optimizers["SGD"] = torch.optim.SGD
        self.optimizers["RMSprop"] = torch.optim.RMSprop
        self.optimizers["ASGD"] = torch.optim.ASGD
        
        self.metric_func = defaultdict(int)
        self.metric_func["KL_sym"] = KL_sym
        self.metric_func["KL"] = KL
        self.metric_func["JSD"] = JSD
        self.metric_func["PE"] = PE
        self.metric_func["PE_sym"] = PE_sym
        self.metric_func["W"] = Wasserstein
    
    def predict(self, X):
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = self.reference_test(X_auto)
        scores = []
        for i in range(len(reference)):
            X_, y_ = self.preprocess(reference[i], test[i])
            score = self.reference_test_predict(X_, y_)
            scores.append(score)
        T_scores = np.array([T[i] for i in range(len(reference))])
        scores = np.array(scores)
        
        res = []
        
        # todo optimize memory
        T_uni = np.arange(len(X))
        T_scores = T_scores - self._time_shift
        unified_score = self.unified_score(T_uni, T_scores, scores)
        avg_unified_score = SMA(unified_score, self.avg_window)
        peaks = res.append(self.find_peaks_cwt(avg_unified_score, widths=self.peak_widths))
        
        return avg_unified_score, peaks
    
    def reference_test(self, X):
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
    
    def preprocess(self, X_ref, X_test):
        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))
        
        X = self.scaler.fit_transform(X)
        
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        return X, y
    
    def unified_score(self, T, T_score, score):
        inter = interpolate.interp1d(T_score, score, kind='previous', fill_value=(0, 0), bounds_error=False)
        uni_score = inter(T)
        return uni_score
    
    def find_peaks_cwt(self, vector, *args, **kwargs):
        peaks = find_peaks_cwt(vector, *args, **kwargs)
        return peaks
    
    @abstractmethod
    def reference_test_predict(self, X_ref, X_test):
        pass


class OnlineNNClassifier(ChangePointDetection):
    def __init__(self, net="default", n_inputs=1, nn_hidden=50, dropout=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.BCELoss()

        self.net = MyNN(n_inputs=n_inputs, n_hidden=nn_hidden, dropout=dropout) if net == "default" else net
        self.opt = self.optimizers[kwargs["optimizer"]](
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.lam
        )  # todo ASGD

    def reference_test_predict(self, X, y):
        self.net.train(False)
        n_last = min(self.window_size, self.step)
        ref_preds = self.net(X[y == 0][-n_last:]).detach().numpy()
        test_preds = self.net(X[y == 1][-n_last:]).detach().numpy()
        
        self.net.train(True)
        for epoch in range(self.n_epochs):  # loop over the dataset multiple times
            # forward + backward + optimize
            outputs = self.net(X)
            loss = self.criterion(outputs.squeeze(), y)
            
            # set gradients to zero
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
        score = self.metric_func[self.metric](ref_preds, test_preds)
        return score


class OnlineNNRuLSIF(ChangePointDetection):
    
    def __init__(self, alpha, n_inputs=10, net="default",
                 nn_hidden=50, dropout=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.net1 = MyNNRegressor(n_inputs=n_inputs, n_hidden=nn_hidden, dropout=dropout) if net == "default" else net
        self.net2 = deepcopy(self.net1)
        
        self.opt1 = self.optimizers[kwargs["optimizer"]](
            self.net1.parameters(),
            lr=self.lr,
            weight_decay=self.lam
        )  # todo ASGD
        
        self.opt2 = self.optimizers[kwargs["optimizer"]](
            self.net2.parameters(),
            lr=self.lr,
            weight_decay=self.lam
        )
    
    def compute_loss(self, y_pred_batch_ref, y_pred_batch_test):
        loss = 0.5 * (1 - self.alpha) * (y_pred_batch_ref ** 2).mean() + \
               0.5 * self.alpha * (y_pred_batch_test ** 2).mean() - (y_pred_batch_test).mean()
        
        return loss
    
    def reference_test_predict(self, X, y):
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
            loss1 = self.compute_loss(y_pred_batch_ref, y_pred_batch_test)
            
            # set gradients to zero
            self.opt1.zero_grad()
            loss1.backward()
            self.opt1.step()
            
            # forward + backward + optimize
            y_pred_batch = self.net2(X).squeeze()
            y_pred_batch_ref = y_pred_batch[y == 1]
            y_pred_batch_test = y_pred_batch[y == 0]
            loss2 = self.compute_loss(y_pred_batch_ref, y_pred_batch_test)
            
            # set gradients to zero
            self.opt2.zero_grad()
            loss2.backward()
            self.opt2.step()
        
        score = (0.5 * np.mean(test_preds) - 0.5) + (0.5 * np.mean(ref_preds) - 0.5)
        return score
