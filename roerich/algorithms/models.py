import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ClassificationNetwork(nn.Module):
    
    def __init__(self, n_inputs=1, n_hidden=10):
        
        super(ClassificationNetwork, self).__init__()
        self.model = nn.Sequential(nn.Linear(n_inputs, n_hidden), 
                                   nn.Tanh(), 
                                   nn.Linear(n_hidden, 1), 
                                   nn.Sigmoid()
                                   )

    def forward(self, x):
        return self.model(x)    
    
    


class NNClassifier(object):
    
    def __init__(self, n_hidden=10, n_epochs=10, batch_size=64, lr=0.01, l2=0.):        
        
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.scaler = StandardScaler()


    def get_data_loader(self, X, y):

        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        data = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        return data_loader

        
    def fit(self, X, y):

        X_ss = self.scaler.fit_transform(X)
        train_loader = self.get_data_loader(X_ss, y)

        self.model = ClassificationNetwork(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = nn.BCELoss()
        
        self.model.train(True)
        for epoch_i in range(self.n_epochs):
            for x_batch, y_batch in train_loader:
                y_pred_batch = self.model(x_batch)
                loss = loss_func(y_pred_batch, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
    
    
    def predict_proba(self, X):
        
        X_ss = self.scaler.transform(X)
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        
        self.model.train(False)
        y_pred = self.model(X_tensor)
        y_pred_1 = y_pred.cpu().detach().numpy()
        y_pred = np.hstack((1-y_pred_1, y_pred_1))
        
        return y_pred
    
    
    
    
# Regreesions
class RegressionNetwork(nn.Module):
    
    def __init__(self, n_inputs=1, n_hidden=10):
        
        super(RegressionNetwork, self).__init__()
        self.model = nn.Sequential(nn.Linear(n_inputs, n_hidden), 
                                   nn.Tanh(), 
                                   nn.Linear(n_hidden, 1)
                                   )

    def forward(self, x):
        return self.model(x)
    
    



class NNRuLSIFRegressor(object):
    
    def __init__(self, n_hidden=10, n_epochs=10, batch_size=64, lr=0.01, l2=0., alpha=0.05):        
        
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.alpha = alpha
        self.scaler = StandardScaler()


    def get_data_loader(self, X, y):

        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        data = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        return data_loader


    def rulsif_loss(self, y_pred, y):

        y_pred_ref = y_pred[y == 0]
        y_pred_test = y_pred[y == 1]
        loss = 0.5 * (1 - self.alpha) * (y_pred_ref**2).mean() + \
               0.5 *      self.alpha  * (y_pred_test**2).mean() - (y_pred_test).mean()

        return loss


    def fit(self, X, y):

        X_ss = self.scaler.fit_transform(X)
        train_loader = self.get_data_loader(X_ss, y)

        self.model = RegressionNetwork(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train(True)
        for epoch_i in range(self.n_epochs):
            for x_batch, y_batch in train_loader:
                y_pred_batch = self.model(x_batch)
                loss = self.rulsif_loss(y_pred_batch, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
        
    
    def predict(self, X):
        
        X_ss = self.scaler.transform(X)
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        
        self.model.train(False)
        y_pred = self.model(X_tensor)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred[y_pred <= 0] *= 0
        
        return y_pred
    
    
    

# GBDT-RuLSIF    
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


class GBDTRuLSIFRegressor(object):
    
    def __init__(self, n_estimators=100, learning_rate=0.1, sample_frac=0.7, alpha=0.1, 
                 max_depth=4, min_samples_leaf=1, min_samples_split=2, splitter='best', 
                 max_features=None):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.sample_frac = sample_frac
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.splitter = splitter
        self.max_features = max_features
        self.estimators = []
        
        
    def pe_loss(self, pred, y):
        L = (-0.5 *       self.alpha *  (pred)**2) * (y == 1) + \
            (-0.5 * (1. - self.alpha) * (pred)**2) * (y == 0) + pred * (y == 1) - 0.5
        return -L
    
    
    def pe_loss_grad(self, pred, y):
        dL = (-1. *       self.alpha  * (pred)) * (y == 1) + \
             (-1. * (1. - self.alpha) * (pred)) * (y == 0) + 1. * (y == 1)
        return -dL
    
    
    def fit(self, X, y):
        for i in range(self.n_estimators):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.sample_frac)
            pred_train = self.predict(X_train)
            grad = - self.pe_loss_grad(pred_train, y_train)
            atree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, 
                                          min_samples_split=self.min_samples_split, splitter=self.splitter, 
                                          max_features=self.max_features)
            atree.fit(X_train, grad)
            self.estimators.append(atree)
        
        
    def predict(self, X):
        predictions = 1. + 0.1 * np.random.rand(len(X))
        for est in self.estimators:
            pred = est.predict(X)
            predictions += self.learning_rate * pred
        return predictions