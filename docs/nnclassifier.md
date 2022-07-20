# NNClassifier

## Description

`NNClassifier` [1] is an algorithm for the direct density ratio estimation for two samples. It is a Neural Network model for binary classification task. The algorithm takes two samples and learns the ratio of their probability densities without estimation of individual distributions. The classifier learns to separate the samples into two classes. According to [1], the density ratio $w(x)$ can be estimated using predictions $f(x)$ of the classifier:

$$
w(x) = \frac{f(x)}{1 - f(x)}
$$

The density ratios help to calculate of different dissimilarity scores between the samples, that are used in change-point detection and other applications.


- [1] Mikhail Hushchyn and Andrey Ustyuzhanin. “Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation.” J. Comput. Sci. 53 (2021): 101385. [[journal]](https://doi.org/10.1016/j.jocs.2021.101385) [[arxiv]](https://doi.org/10.48550/arXiv.2001.06386)

## Usage

```python
from roerich.algorithms import NNClassifier
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
import numpy as np

N = 1000

# generate samples
p0 = uniform(-5, 10)
p1 = norm(0, 1)
X = np.concatenate((p0.rvs((N, 1)), p1.rvs((N, 1))))
y = np.array([0]*N + [1]*N)

# true density ratio
true_ratio = p1.pdf(X) / p0.pdf(X)

# direct density ratio estimation
reg = NNClassifier(n_hidden=10, n_epochs=100, batch_size=64, lr=0.01, l2=0.001)
reg.fit(X, y)
pred_ratio = reg.predict_proba_ratio(X)

# visualization
plt.scatter(X, pred_ratio, label='pred')
plt.scatter(X, true_ratio, label='true')
plt.legend()
plt.show()
```
