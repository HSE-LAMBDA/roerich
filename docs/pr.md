# Precision and Recall

## Precision and Recall

Consider a time series with $n$ change-points at moments $\tau_{1}$, $\tau_{2}$, ..., $\tau_{n}$. Suppose that an algorithm recognises $m$ change points at moments $\hat{\tau}_{1}$, $\hat{\tau}_{2}$, ..., $\hat{\tau}_{m}$. Following [1], a set of correctly detected change-points is defined as True Positive (TP):

$$
    \text{TP} = \{ \tau_{i} | \exists \hat{\tau}_{j}: |\hat{\tau}_{j} - \tau_{i}| < M \}
$$

where $M$ is a margin size, maximum distance allowed between true and predicted change points. Then, Precision and Recall metrics are calculated as follows:

$$
    \text{Precision} = \frac{|\text{TP}|}{m}
$$

$$
    \text{Recall} = \frac{|\text{TP}|}{n}
$$

[1] C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. Signal Processing, 167:107299, 2020. [[journal]](https://doi.org/10.1016/j.sigpro.2019.107299)

## PR curve

Suppose that we know detection scores $s_{1}$, $s_{2}$, ..., $s_{m}$ for all $m$ recognised change points. Let's calculate Precision and Recall metrics for different threshold values $\nu$ for the scores:

$$
    \text{TP}(\nu) = \{ \tau_{i} | \exists \hat{\tau}_{j}: |\hat{\tau}_{j} - \tau_{i}| < M \text{ and } (s_j \ge \nu) \}
$$

$$
    m(\nu) = \sum_{j: s_j \ge \nu} 1
$$

$$
    \text{Precision}(\nu) = \frac{|\text{TP}(\nu)|}{m(\nu)}
$$

$$
    \text{Recall}(\nu) = \frac{|\text{TP}(\nu)|}{n}
$$

PR curve is the dependency of Precision$(\nu)$ form Recall$(\nu)$.

## Usage

```python
from roerich.metrics import precision_recall_scores, precision_recall_curve, auc_score
import matplotlib.pyplot as plt

cps_true = [100, 200, 300, 400, 500]
cps_pred = [105, 230, 310, 350, 405, 490]
cps_score_pred = [1, 2, 3, 0.1, 5, 6]

# precision and recall
precision, recall = precision_recall_scores(cps_true, cps_pred, window=20)
print('Precision: ', precision)
print('Recall: ', recall)

# PR curve and AUC
thr, precision, recall = precision_recall_curve(cps_true, cps_pred, cps_score_pred, window=20)
auc = auc_score(thr, precision, recall)
print("PR AUC: ", auc)

# visualization
plt.plot(recall, precision)
plt.show()
```
