# Roerich

`roerich` is a library for online and offline change point detection. Currently, it implements 
algorithms based on direct density estimation from this article:

> Hushchyn, Mikhail, and Andrey Ustyuzhanin. ‘Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation’. ArXiv:2001.06386 [Cs, Stat], Jan. 2020. arXiv.org, http://arxiv.org/abs/2001.06386.

## Dependencies and install

## Basic usage 

Make sure that your data has a shape `(seq_len, n_dims)` or you can generate synthetic data:
```python
import numpy as np
import roerich
 
X, label = roerich.generate_dataset(period=2000, N_tot=20000)
T = np.arange(len(X))
```

You can use two algorithms: `CLF` or `RuLSIF`: 

```python
cpd = roerich.OnlineNNClassifier(net='default', scaler="default", metric="KL_sym",
                  periods=1, window_size=10, lag_size=500, step=10, n_epochs=10,
                  lr=0.1, lam=0.0001, optimizer="Adam"
                 )

# Detect change points
score, peaks = cpd.predict(X)
```

For data visualization use: 
```python
roerich.display(X, T, label, score, T, peaks)
```
![](images/demo.png)

## Changelog

See the [changelog](https://github.com/HSE-LAMBDA/roerich/blob/master/CHANGELOG.md) for a history of notable changes to roerich.

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