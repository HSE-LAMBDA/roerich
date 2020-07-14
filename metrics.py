import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


def autoregression_matrix(X, periods=1, fill_value=0):
    shifted_x = [pd.DataFrame(X).shift(periods=i, fill_value=fill_value).values for i in range(periods)]
    X_auto = np.hstack(tuple(shifted_x))
    return X_auto


def KL_score_unsym(ref_ratios, test_ratios):
    score = np.mean(np.log(test_ratios))
    return score


def KL_score(ref_ratios, test_ratios):
    score = KL_score_unsym(ref_ratios, test_ratios) + KL_score_unsym(1. / test_ratios, 1. / ref_ratios)
    return score


def PE_score_unsym(ref_ratios, test_ratios, alpha=0.):
    score = (-0.5 * alpha * np.mean(test_ratios ** 2)) + \
            (-0.5 * (1. - alpha) * np.mean(ref_ratios ** 2)) + np.mean(test_ratios) - 0.5
    return score


def PE_score(ref_ratios, test_ratios, alpha=0.):
    score = PE_score_unsym(ref_ratios, test_ratios, alpha)  # - PE_score_unsym(test_ratios, ref_ratios, alpha)
    return score


def KL(ref_preds, test_preds):
    return np.mean(np.log(test_preds + 10 ** -3)) - np.mean(np.log(1. - test_preds + 10 ** -3))


def KL_sym(ref_preds, test_preds):
    return np.mean(np.log(test_preds + 10 ** -3)) - np.mean(np.log(1. - test_preds + 10 ** -3)) + \
           np.mean(np.log(1. - ref_preds + 10 ** -3)) - np.mean(np.log(ref_preds + 10 ** -3))


def JSD(ref_preds, test_preds):
    return np.log(2) + 0.5 * np.mean(np.log(test_preds + 10 ** -3)) + 0.5 * np.mean(np.log(1. - ref_preds + 10 ** -3))


def PE(ref_preds, test_preds):
    scores = test_preds / (1. - test_preds + 10 ** -6) - 1.
    scores = np.clip(scores, 0, 1000)
    return np.mean(scores)


def PE_sym(ref_preds, test_preds):
    scores_1 = test_preds / (1. - test_preds + 10 ** -6) - 1.
    scores_1 = np.clip(scores_1, 0, 1000)
    scores_2 = (1. - ref_preds) / (ref_preds + 10 ** -6) - 1.
    scores_2 = np.clip(scores_2, 0, 1000)
    return np.mean(scores_1) + np.mean(scores_2)


def Wasserstein(ref_preds, test_preds):
    return np.mean(test_preds) - np.mean(ref_preds)
