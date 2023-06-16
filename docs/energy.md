# EnergyDistanceCalculator

## Description

`EnergyDistanceCalculator` is an offline change point detection algorithm based on energy distance [1]. It scans a signal with a two consecutive windows with some width. The method calculates the energy distance for observations inside these windows, and considers it as a change point detection score for the windows pair.


- [1] Szekely, Gabor. (2003). E-Statistics: The energy of statistical samples. 


## Usage

```python
import roerich
from roerich.change_point import EnergyDistanceCalculator

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# change points detection
cpd = EnergyDistanceCalculator(window_size=100, step=1)
score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```
