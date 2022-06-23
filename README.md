# Welcome to Roerich

[![PyPI version](https://badge.fury.io/py/roerich.svg)](https://badge.fury.io/py/roerich)
[![Downloads](https://pepy.tech/badge/roerich)](https://pepy.tech/project/roerich)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

`Roerich` is a python library for online and offline change point detection in time series data. It was named after the painter Nicholas Roerich, known as the Master of the Mountains. Read more at: https://www.roerich.org.

![](images/700125v1.jpeg)
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
python setup.py install
```

## Basic usage

The following code snippet generates a noisy synthetic data, performs change point detection, and displays the results. If you use own dataset, make
sure that it has a shape `(seq_len, n_dims)`.
```python
import roerich
from roerich.algorithms import ChangePointDetectionClassifier

# generate time series
X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

# detection
cpd = ChangePointDetectionClassifier()
score, cps_pred = cpd.predict(X)

# visualization
roerich.display(X, cps_true, score, cps_pred)
```

![](images/demo.png)

## Related libraries

[![Generic badge](https://img.shields.io/badge/^.^-ruptures-blue.svg)](https://github.com/deepcharles/ruptures)
[![Generic badge](https://img.shields.io/badge/^.^-klcpd-blue.svg)](https://github.com/HolyBayes/klcpd)
[![Generic badge](https://img.shields.io/badge/^.^-tire-blue.svg)](https://github.com/HolyBayes/TIRE_pytorch)
[![Generic badge](https://img.shields.io/badge/^.^-bocpd-blue.svg)](https://github.com/hildensia/bayesian_changepoint_detection)

## Thanks to all our contributors

<a href="https://github.com/HSE-LAMBDA/roerich/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=HSE-LAMBDA/roerich" />
</a>

## License

```
BSD 2-Clause License

Copyright (c) 2017, ENS Paris-Saclay, CNRS
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
