from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.signal import find_peaks_cwt

from .metrics import autoregression_matrix
from .metrics import KL_sym, KL, JSD, PE, PE_sym, Wasserstein
from .net import MyNN
from .scaler import SmaScalerCache


class ChangePointDetection(metaclass=ABCMeta):
    def __init__(self, n_inputs, net="default", scaler="default", metric="KL", window_size=1, periods=10, lag_size=0,
                 step=1, n_epochs=100, lr=0.01, lam=0, optimizer="Adam", debug=0,
                 nn_hidden=50, dropout=0, find_peaks=False, peak_widths=[50], plot_peak_height=10,
                 shift=False, unify=False, average=False, avg_window=1):
        
        self.net = MyNN(n_inputs=n_inputs, n_hidden=nn_hidden, dropout=dropout) if net == "default" else net
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
        
        self.shift = shift
        self.unify = unify
        self._time_shift = lag_size + window_size

        self.find_peaks = find_peaks
        self.peak_widths = peak_widths
        self.plot_peak_height = plot_peak_height
        
        self.average = average
        self.avg_window = avg_window
        
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
            X, y = self.preprocess(reference[i], test[i])
            score = self.reference_test_predict(X, y)
            scores.append(score)
        T_scores = np.array([T[i] for i in range(len(reference))])
        scores = np.array(scores)
        
        res = []
        
        if self.unify and self.shift:
            T_scores = T_scores - self._time_shift
            res = [T - self._time_shift, self.unified_score(T, T_scores, scores)]
        elif self.unify:
            res = [T, self.unified_score(T, T_scores, scores)]
        elif self.shift:
            res = [T_scores - self._time_shift, scores]
        else:
            res = [T_scores, scores]
        
        if self.find_peaks:
            res.append(self.find_peaks_cwt(res[-1], widths=self.peak_widths))
        
        return res
    
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
    
    def display(self, X, T, L, S, Ts, peaks=None, s_max=10):
        n = X.shape[1] + 1 if peaks is None else X.shape[1] + 2
    
        plt.figure(figsize=(12, n*2.5+0.25))
    
        for i in range(X.shape[1]):
        
            plt.subplot(n, 1, i+1)
            ax = X[:, i]
            plt.plot(T, ax, linewidth=2, label='Original signal', color='C0')
            for t in T[L == 1]:
                plt.plot([t]*2, [ax.min(), ax.max()], color='0', linestyle='--')
            plt.ylim(ax.min(), ax.max())
            plt.xlim(0, T.max())
            plt.xticks(size=16)
            plt.yticks(size=16)
            plt.legend(loc='upper left', fontsize=16)
            plt.tight_layout()
    
        score_plot_ix = n if peaks is None else n - 1
        plt.subplot(n, 1, score_plot_ix)
        plt.plot(Ts, S, linewidth=3, label="Change-point score", color='C3')
        for t in T[L == 1]:
            plt.plot([t]*2, [-1, s_max], color='0', linestyle='--')
        
        # display find peaks #todo refactoring
        if peaks:
            plt.subplot(n, 1, n)
            new_score_peaks = np.zeros(len(T))
            new_score_peaks[peaks] = self.plot_peak_height
            plt.plot(new_score_peaks, S, linewidth=3, label="Peaks", color='C4')
            for t in T[L == 1]:
                plt.plot([t]*2, [-1, s_max], color='0', linestyle='--')
        
        plt.ylim(-1, s_max)
        plt.xlim(0, T.max())
        plt.xticks(size=16)
        plt.yticks(np.arange(0, s_max+1, 5), size=16)
        plt.xlabel("Time", size=16)
        plt.legend(loc='upper left', fontsize=16)
        plt.tight_layout()
        

if __name__ == '__main__':
    pass
