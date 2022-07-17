# Change point

## Formal definition

Abrupt change of time series behaviour is called a change-point. Consider a $d-$dimensional signal with a change point at a moment $\nu$:

$$
x_1, x_2, \cdots, x_{\nu-1}, x_{\nu}, x_{\nu+1}, x_{\nu+2}, \cdots
$$

In the most general case, the observations are arbitrary dependent and nonidentically distributed. Change point is a moment of time, when distribution of time series observations changes. Thus, it can be described as changing of conditional pre-change densities $p_0(x_i|x_1, \cdots, x_{i-1})$ for $i \le \nu$ to conditional post-change densities $p_1(x_i|x_1, \cdots, x_{i-1})$ for $i > \nu$. In a typical setup this is an unsupervised task with unknown signal distributions. Figure 1 shows an example with several change points and results of their detection.

![](https://raw.githubusercontent.com/HSE-LAMBDA/roerich/main/images/demo.png)
_Figure 1. Example of change points._


## Applications

There is a range of various applications of change point detection [1]: quality control of production process, structural health monitoring of wind turbines, and aircraft, detecting multiple sensor faults, detecting road traffic incidents or changes in highway traffic condition, chemical process control, physiological data analysis, surveillance of daily disease counts, nanoscale analysis of soft biomaterials, biosurveillance, radio-astronomy and interferometry, spectrum sensing in cognitive radio systems, leak detection in water channels, environmental monitoring, handling climate changes, navigation systems monitoring, human motion analysis, video scene analysis, sequential steganography, biometric identification, onset detection in music signals, detecting changes in large payment card datasets, distributed systems monitoring, detection of intrusion, viruses, and other denial of service (DoS) attacks, segmentation of signals and images, tracking the preferences of users in recommendation systems, seismic data processing, analysis of financial data and others.

[1] Alexander Tartakovsky, Igor Nikiforov, and Michele Basseville. 2014. Sequential Analysis: Hypothesis Testing and Changepoint Detection (1st. ed.). Chapman & Hall/CRC.
