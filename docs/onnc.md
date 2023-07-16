# OnlineNNClassifier

## Description

`OnlineNNClassifier` [1] is an online change point detection algorithm based on a binary classifier. It scans a signal with a two windows with small width (1, 5, 10) and a lag (50, 100) between them. The method uses a neural network classifier for direct estimation of the probability density ratio for observations inside these windows. We train the NN in online manner: update its weights with new observations of the signal. Then, it calculates the symmetrical Kullback-Leibler (KL_sym) divergence based on these ratios and considers it as a change point detection score for the windows pair.


- [1] Mikhail Hushchyn, Kenenbek Arzymatov and Denis Derkach. “Online Neural Networks for Change-Point Detection.” ArXiv abs/2010.01388 (2020). [[arxiv]](https://doi.org/10.48550/arXiv.2010.01388)

## Usage

```python
import roerich
from roerich.change_point import OnlineNNClassifier

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# change points detection
cpd = OnlineNNClassifier(periods=1, window_size=1, lag_size=100, step=1,
                         n_epochs=1, lr=0.01, lam=0.0001, optimizer="Adam")

score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```

## Usage with custom network

```python
import torch.nn as nn

# custom network
class MyNN(nn.Module):
    def __init__(self, n_inputs=1):
        super(MyNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, 100), 
                                 nn.ReLU(), 
                                 nn.Linear(100, 100), 
                                 nn.ReLU(), 
                                 nn.Linear(100, 1), 
                                 nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


# change points detection
cpd = OnlineNNClassifier(net=MyNN, periods=1, window_size=1, lag_size=100, step=1,
                         n_epochs=1, lr=0.001, lam=0.0001, optimizer="Adam")

score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```
