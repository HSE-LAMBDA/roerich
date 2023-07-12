# ChangePointDetectionClassifier

## Description

`ChangePointDetectionClassifier` [1] is an offline change point detection algorithm based on binary classifiers. It scans a signal with a two consecutive windows with some width. The method uses a binary classifier for direct estimation of the probability density ratio for observations inside these windows. Then, it calculates the symmetrical Kullback-Leibler (KL_sym) divergence based on these ratios and considers it as a change point detection score for the windows pair.


- [1] Mikhail Hushchyn and Andrey Ustyuzhanin. “Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation.” J. Comput. Sci. 53 (2021): 101385. [[journal]](https://doi.org/10.1016/j.jocs.2021.101385) [[arxiv]](https://doi.org/10.48550/arXiv.2001.06386)

## Usage

```python
import roerich
from roerich.change_point import ChangePointDetectionClassifier

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# change points detection
# base_classifier = 'logreg', 'qda', 'dt', 'rf', 'mlp', 'knn', 'nb'
# metric = 'klsym', 'pesym', 'jsd', 'mmd', 'fd'
cpd = ChangePointDetectionClassifier(base_classifier='mlp', metric='klsym', periods=1,
                                     window_size=100, step=1, n_runs=1)
score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```

## Usage with custom classifier

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# sklearn-like binary classifier
clf = QuadraticDiscriminantAnalysis()

# change points detection
cpd = ChangePointDetectionClassifier(base_classifier=clf, metric='klsym', periods=1,
                                     window_size=100, step=1, n_runs=1)
score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```
