from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Union, Type, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks_cwt

from roerich.algorithms.net import MyNN, MyNNRegressor
from roerich.utils import autoregression_matrix, unified_score
from roerich.metrics.metrics import KL_sym, KL, JSD, PE, PE_sym, Wasserstein
from roerich.algorithms.scaler import SmaScalerCache
from roerich.helper import SMA


class ChangePointDetection(metaclass=ABCMeta):
    def __init__(self, scaler: Any = "default", metric: str = "KL", window_size: int = 1, periods: int = 10,
                 lag_size: int = 0, step: int = 1, n_epochs: int = 100, lr: float = 0.01, lam: float = 0,
                 optimizer: str = "Adam", debug: int = 0):
        """
        
        Parameters
        ----------
        scaler: A scaler object is used to scale an input data. The default one is `SmaScalerCache`
        metric: A loss function during optimize step of NN. Can be one of the following KL_sym, KL, JSD, PE, PE_sym, Wasserstein
        window_size: A size of a window when splitting input data into train and test arrays
        periods: A number of previous data-points used when constructing autoregressive matrix
        lag_size: A distance between train- and test- windows
        step: Each `step`-th data-point is used when creating the input dataset
        n_epochs: A number of epochs during training NN
        lr: A learning rate at each step of optimizer
        lam: A regularization rate
        optimizer: One of Adam, SGD, RMSprop or ASGD optimizers
        debug: default zero
        """
        
        self.scaler = SmaScalerCache(window_size + lag_size) if scaler == "default" else scaler
        
        self.metric = metric
        self.window_size = window_size
        self.periods = periods
        self.lag_size = lag_size
        self.step = step
        self.n_epochs = n_epochs
        self.lr = lr
        self.lam = lam
        self.debug = debug
        
        self._time_shift = lag_size + window_size
        self.avg_window = lag_size + window_size
        self.peak_widths = [0.25 * (lag_size + window_size)]
        
        self.optimizers = defaultdict(lambda: torch.optim.Adam)
        
        self.optimizers["Adam"] = torch.optim.Adam
        self.optimizers["SGD"] = torch.optim.SGD
        self.optimizers["RMSprop"] = torch.optim.RMSprop
        self.optimizers["ASGD"] = torch.optim.ASGD
        self.optimizer = self.optimizers[optimizer]
        
        self.metric_func = {"KL_sym": KL_sym,
                            "KL": KL,
                            "JSD": JSD,
                            "PE": PE,
                            "PE_sym": PE_sym,
                            "W": Wasserstein
                            }
    
    @abstractmethod
    def init_net(self, n_inputs: int) -> None:
        """
        Initialize neural network based on `self.base_net` class
        Parameters
        ----------
        n_inputs: Number of inputs of neural network
        -------

        """
        pass
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Tuple[Any, Any]:
        """
        Determines a CPD score for every data-point
        Parameters
        ----------
        X: An input data

        Returns `avg_unified_score`: An averaged, unified and shifted CPD score for every data-point in X
                `peaks` Locations of CPD points along all data-points
        -------

        """
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        self.init_net(X_auto.shape[1])
        T, reference, test = self.reference_test(X_auto)
        scores = []
        for i in range(len(reference)):
            X_, y_ = self.preprocess(reference[i], test[i])
            score = self.reference_test_predict(X_, y_)
            scores.append(score)
        T_scores = np.array([T[i] for i in range(len(reference))])
        scores = np.array(scores)
        
        # todo optimize memory
        T_uni = np.arange(len(X))
        T_scores = T_scores - self._time_shift
        un_score = unified_score(T_uni, T_scores, scores)
        avg_unified_score = SMA(un_score, self.avg_window)
        peaks = self.find_peaks_cwt(avg_unified_score, widths=self.peak_widths)
        
        return avg_unified_score, peaks
    
    def reference_test(self, X: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates reference and test datasets based on autoregressive matrix.
        
        Parameters
        ----------
        X: An autoregressive matrix

        Returns tuple of numpy arrays: time-steps, reference and test datasets
        -------

        """
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
    
    def preprocess(self, X_ref: np.ndarray, X_test: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates X and y datasets for training NN by stacking reference and test datasets.
        Also applies a scaling transformation into resulting X dataset.
        Labels for reference data-points is 1s.
        Labels for test data-points is 0s.
        
        Parameters
        ----------
        X_ref: reference data-points
        X_test: test data-points

        Returns
        -------
        Tuple of training data
        """
        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))
        
        X = self.scaler.fit_transform(X)
        
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        return X, y
    
    def find_peaks_cwt(self, vector, *args, **kwargs):
        """
        Find peaks function based on scipy.signal package
        Parameters
        ----------
        vector: CPD scores array
        args: see docs for https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html
        kwargs

        Returns
        -------
        Array with location of peaks
        """
        peaks = find_peaks_cwt(vector, *args, **kwargs)
        return peaks
    
    @abstractmethod
    def reference_test_predict(self, X: torch.Tensor, y: torch.Tensor):
        """
        Training process of forward, backward and optimize steps.
        Parameters
        ----------
        X: train data
        y: train labels

        Returns
        -------
        None
        """
        pass


class OnlineNNClassifier(ChangePointDetection):
    def __init__(self, net: Union[Type[nn.Module], str] = "default", *args, **kwargs):
        """
        Parameters
        ----------
        net: Custom torch.nn.Module neural network or "default" one
        args: see parent class
        kwargs: see parent class
        """
        super().__init__(*args, **kwargs)
        self.criterion = nn.BCELoss()
        
        self.base_net = MyNN if net == "default" else net
        self.net = None
        self.opt = None
    
    def init_net(self, n_inputs):
        self.net = self.base_net(n_inputs)
        self.opt = self.optimizer(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.lam
        )
    
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
    
    def __init__(self, alpha, net="default", *args, **kwargs):
        """
        Parameters
        ----------
        alpha: The `alpha` parameter in a loss function
        net: Custom torch.nn.Module neural network or "default" one
        args: see parent class
        kwargs: see parent class
        """
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.base_net = MyNNRegressor if net == "default" else net
        self.net1 = None
        self.net2 = None
        self.opt1 = None
        self.opt2 = None
    
    def init_net(self, n_inputs):
        self.net1 = self.base_net(n_inputs)
        self.opt1 = self.optimizer(
            self.net1.parameters(),
            lr=self.lr,
            weight_decay=self.lam
        )
        self.net2 = deepcopy(self.net1)
        self.opt2 = deepcopy(self.opt1)
    
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
