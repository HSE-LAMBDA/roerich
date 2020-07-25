from copy import deepcopy

import numpy as np
import torch.nn as nn

from .cpd import ChangePointDetection


class ChangePointDetectionOnlineCLF(ChangePointDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.BCELoss()
        
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


class ChangePointDetectionOnlineRuLSIF(ChangePointDetection):
    
    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.net1 = self.net
        self.net2 = deepcopy(self.net)
        
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
