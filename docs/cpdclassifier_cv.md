# ChangePointDetectionClassifierCV

## Description

`ChangePointDetectionClassifierCV` is a version of `ChangePointDetectionClassifier` [1] method with K-Fold cross-validation.

- [1] Mikhail Hushchyn and Andrey Ustyuzhanin. “Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation.” J. Comput. Sci. 53 (2021): 101385. [[journal]](https://doi.org/10.1016/j.jocs.2021.101385) [[arxiv]](https://doi.org/10.48550/arXiv.2001.06386)

## Usage

```python
import roerich
from roerich.change_point import ChangePointDetectionClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# sklearn-like binary classifier
clf = QuadraticDiscriminantAnalysis()

# change points detection
cpd = ChangePointDetectionClassifierCV(base_classifier=clf, metric='KL_sym', periods=1,
                                       window_size=100, step=1, n_splits=5)
score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```
