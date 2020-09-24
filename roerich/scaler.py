import numpy as np
import collections
from abc import ABCMeta, abstractmethod


class BaseScaler(metaclass=ABCMeta):
    def __init__(self, N):
        self.N = N
        self.mu = None
        self.sigma = None
    
    @abstractmethod
    def fit(self, X):
        pass
    
    def transform(self, X):
        return (X - self.mu) / self.sigma
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class SmaScaler(BaseScaler):
    def fit(self, X):
        if self.sigma is None:
            self.mu = np.mean(X, axis=0)
            self.sigma = np.std(X, axis=0)
        else:
            x = np.mean(X, axis=0)
            self.sigma = np.sqrt(self.sigma ** 2 * (self.N - 2) / (self.N - 1) + (x - self.mu) ** 2 / self.N)
            self.mu += (x - self.mu) / self.N


class EmaScaler(BaseScaler):
    def __init__(self, N, smooth):
        super().__init__(N)
        self.alpha = smooth / (1. + self.N)
    
    def fit(self, X):
        if self.sigma is None:
            self.mu = np.mean(X, axis=0)
            self.sigma = np.std(X, axis=0)
        else:
            x = np.mean(X, axis=0)
            self.sigma = (1 - self.alpha) * (self.sigma + self.alpha * (x - self.mu) ** 2)
            self.mu = self.alpha * x + (1 - self.alpha) * self.mu


class SmaScalerCache(BaseScaler):
    def __init__(self, N):
        super(SmaScalerCache, self).__init__(N)
        self.cache = collections.deque()
    
    def fit(self, X):
        if len(self.cache) + len(X) > self.N:
            over = len(self.cache) + len(X) - self.N
            print(over)
            print(self.cache[-over:])
            self.cache = self.cache[-over:]
        self.cache.extend(X)
        
        self.mu = np.mean(self.cache, axis=0)
        self.sigma = np.std(self.cache, axis=0)


class EmaScalerCache(EmaScaler):  # todo
    def __init__(self, N, smooth):
        super().__init__(N, smooth)
        self.cache = collections.deque()
    
    def fit(self, X):
        if self.sigma is None:
            self.mu = np.mean(X, axis=0)
            self.sigma = np.std(X, axis=0)
        else:
            x = np.mean(X, axis=0)
            self.sigma = (1 - self.alpha) * (self.sigma + self.alpha * (x - self.mu) ** 2)
            self.mu = self.alpha * x + (1 - self.alpha) * self.mu


class MockScaler(BaseScaler):
    def fit(self, X):
        pass
    
    def transform(self, X):
        return X
