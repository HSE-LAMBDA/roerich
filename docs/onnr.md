# OnlineNNRuLSIF

## Description

`OnlineNNRuLSIF` [1] is an online change point detection algorithm based on a RuLSIF [2, 3] regressor. It scans a signal with a two windows with small width (1, 5, 10) and a lag (50, 100) between them. The method uses a neural network RuLSIF regressor for direct estimation of the probability density ratio for observations inside these windows. We train the NN in online manner: update its weights with new observations of the signal. Then, it calculates the Pearson (PE) divergence based on these ratios and considers it as a change point detection score for the windows pair.


- [1] Mikhail Hushchyn, Kenenbek Arzymatov and Denis Derkach. “Online Neural Networks for Change-Point Detection.” ArXiv abs/2010.01388 (2020). [[arxiv]](https://doi.org/10.48550/arXiv.2010.01388)
- [2] Makoto Yamada, Taiji Suzuki, Takafumi Kanamori, Hirotaka Hachiya, Masashi Sugiyama; Relative Density-Ratio Estimation for Robust Distribution Comparison. Neural Comput 2013; 25 (5): 1324–1370. [[DOI]](https://doi.org/10.1162/NECO_a_00442)
- [3] Song Liu, Makoto Yamada, Nigel Collier, Masashi Sugiyama, Change-point detection in time-series data by relative density-ratio estimation, Neural Networks, V. 43, 2013, pp. 72-83.
[[DOI]](https://doi.org/10.1016/j.neunet.2013.01.012)

## Usage

```python
import roerich
from roerich.algorithms import OnlineNNRuLSIF

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# change points detection
cpd = OnlineNNRuLSIF(alpha=0.2, net='default', scaler="default",
                     periods=1, window_size=10, lag_size=100, step=5,
                     n_epochs=10, lr=0.01, lam=0.001, optimizer="Adam")

score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```
