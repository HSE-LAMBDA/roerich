import numpy as np
import pytest
from roerich import metrics

labels_based = [metrics.false_positive_rate,
                metrics.true_positive_rate,
                metrics.precision_recall_scores,
                metrics.precision_score,
                metrics.recall_score]

@pytest.mark.parametrize("algo", labels_based)
def test_labels_based(algo):
    # generate samples
    cps_true = [100, 200, 300, 400, 500]
    cps_pred = [105, 230, 310, 350, 405, 490]

    # detection
    algo(cps_true, cps_pred)



score_based = [metrics.precision_recall_curve,
                metrics.pr_auc]

@pytest.mark.parametrize("algo", score_based)
def test_score_based(algo):
    # generate samples
    cps_true = [100, 200, 300, 400, 500]
    cps_pred = [105, 230, 310, 350, 405, 490]
    cps_score_pred = [1, 2, 3, 0.1, 5, 6]

    # detection
    algo(cps_true, cps_pred, cps_score_pred)
