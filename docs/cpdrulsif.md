# ChangePointDetectionRuLSIF

## Description

`ChangePointDetectionRuLSIF` [1] is an offline change point detection algorithm based on RuLSIF [2, 3] regressors. It scans a signal with a two consecutive windows with some width. The method uses a RuLSIF regression for direct estimation of the probability density ratio for observations inside these windows. Then, it calculates the Pearson (PE) divergence based on these ratios and considers it as a change point detection score for the windows pair.


- [1] Mikhail Hushchyn and Andrey Ustyuzhanin. “Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation.” J. Comput. Sci. 53 (2021): 101385. [[journal]](https://doi.org/10.1016/j.jocs.2021.101385) [[arxiv]](https://doi.org/10.48550/arXiv.2001.06386)
- [2] Makoto Yamada, Taiji Suzuki, Takafumi Kanamori, Hirotaka Hachiya, Masashi Sugiyama; Relative Density-Ratio Estimation for Robust Distribution Comparison. Neural Comput 2013; 25 (5): 1324–1370. [[DOI]](https://doi.org/10.1162/NECO_a_00442)
- [3] Song Liu, Makoto Yamada, Nigel Collier, Masashi Sugiyama, Change-point detection in time-series data by relative density-ratio estimation, Neural Networks, V. 43, 2013, pp. 72-83.
[[DOI]](https://doi.org/10.1016/j.neunet.2013.01.012)

## Usage

```python
import roerich
from roerich.change_point import ChangePointDetectionRuLSIF

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# change points detection
cpd = ChangePointDetectionRuLSIF(periods=1, window_size=100, step=5, n_runs=1)
score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```
