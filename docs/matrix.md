# MatrixImportance

## Description

`MatrixImportance` calculates a matrix of importance for each pair of input features. It helps to understand impact of the features into the change point detection score.

## Usage

```python
import numpy as np
from roerich.explanation import MatrixImportance
from roerich.change_point import EnergyDistanceCalculator

# generate time series
n = 400
X1 = np.random.normal([0, 0, 0], 1, size=(n, 3))
X2 = np.random.normal([0, 1, 2], 1, size=(n, 3))
X = np.concatenate((X1, X2), axis=0)
cps_true = [n]

# base detection algo
cp = EnergyDistanceCalculator(window_size=100)

# importance
mi = MatrixImportance(cp)
# change point on all input features
score, _ = mi.predict(X)
# change point on each pair of features
mat_u = mi.predict_union(X)
# change point with excluding pairs of features
mat_e = mi.predict_exclude(X)

# visualization
roerich.display(X, cps_true, mat_u[:, 1, 2], None)
```
