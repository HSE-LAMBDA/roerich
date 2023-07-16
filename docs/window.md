# SlidingWindows

## Description

`SlidingWindows` is an offline change point detection algorithm based on discrepancy measures [1]. It scans a signal with a two consecutive windows with some width. The method calculates discrepancies between observations inside these windows, and considers it as a change point detection score for the windows pair.

Implemented discrepancy measures:

- Energy distance ('energy')
- Frechet distance ('fd')
- Maximum Mean Discrepancy ('mmd')


[1] C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. Signal Processing, 167:107299, 2020.

## Usage

```python
import roerich
from roerich.change_point import SlidingWindows

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# change points detection
# metric = 'fd', 'mmd', 'energy'
cpd = SlidingWindows(metric='mmd', window_size=100)
score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```
