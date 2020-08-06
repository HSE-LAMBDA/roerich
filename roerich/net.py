import torch.nn as nn


class MyNN(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10, dropout=0.):
        super(MyNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(n_hidden, 1)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class MyNNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10, dropout=0):
        super(MyNNRegressor, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x