# Welcome to Roerich

[![PyPI version](https://badge.fury.io/py/roerich.svg)](https://badge.fury.io/py/roerich)
[![Documentation](https://img.shields.io/badge/documentation-yes-green.svg)](https://hse-lambda.github.io/roerich)
[![Downloads](https://pepy.tech/badge/roerich)](https://pepy.tech/project/roerich)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

`Roerich` is a python library for online and offline change point detection for time series analysis, signal processing, and segmentation. It was named after the painter Nicholas Roerich, known as the Master of the Mountains. Read more at: https://www.roerich.org.

![](https://raw.githubusercontent.com/HSE-LAMBDA/roerich/main/images/700125v1.jpeg)
_Fragment of "Himalayas", 1933_

Currently, the library contains official implementations of change point detection algorithms based on direct density ratio estimation from the following articles:

- Mikhail Hushchyn and Andrey Ustyuzhanin. “Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation.” J. Comput. Sci. 53 (2021): 101385. [[journal]](https://doi.org/10.1016/j.jocs.2021.101385) [[arxiv]](https://doi.org/10.48550/arXiv.2001.06386)
- Mikhail Hushchyn, Kenenbek Arzymatov and Denis Derkach. “Online Neural Networks for Change-Point Detection.” ArXiv abs/2010.01388 (2020). [[arxiv]](https://doi.org/10.48550/arXiv.2010.01388)

## Dependencies and install

```
pip install roerich
```
or
```python
git clone https://github.com/HSE-LAMBDA/roerich.git
cd roerich
pip install -e .
```

## Basic usage

(See more examples in the [documentation](https://hse-lambda.github.io/roerich).)

The following code snippet generates a noisy synthetic data, performs change point detection, and displays the results. If you use own dataset, make
sure that it has a shape `(seq_len, n_dims)`.
```python
import roerich
from roerich.change_point import ChangePointDetectionClassifier

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# detection
# base_classifier = 'logreg', 'qda', 'dt', 'rf', 'mlp', 'knn', 'nb'
# metric = 'klsym', 'pesym', 'jsd', 'mmd', 'fd'
cpd = ChangePointDetectionClassifier(base_classifier='mlp', metric='klsym', window_size=100)
score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```

![](https://raw.githubusercontent.com/HSE-LAMBDA/roerich/main/images/demo.png)

## Support

- Home: [https://github.com/HSE-LAMBDA/roerich](https://github.com/HSE-LAMBDA/roerich)
- Documentation: [https://hse-lambda.github.io/roerich](https://hse-lambda.github.io/roerich)
- For any usage questions, suggestions and bugs use the [issue page](https://github.com/HSE-LAMBDA/roerich/issues), please.

## Related libraries

[![Generic badge](https://img.shields.io/badge/^.^-ruptures-blue.svg)](https://github.com/deepcharles/ruptures)
[![Generic badge](https://img.shields.io/badge/^.^-klcpd-blue.svg)](https://github.com/HolyBayes/klcpd)
[![Generic badge](https://img.shields.io/badge/^.^-tire-blue.svg)](https://github.com/HolyBayes/TIRE_pytorch)
[![Generic badge](https://img.shields.io/badge/^.^-bocpd-blue.svg)](https://github.com/hildensia/bayesian_changepoint_detection)

## Thanks to all our contributors

<a href="https://github.com/HSE-LAMBDA/roerich/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=HSE-LAMBDA/roerich" />
</a>

