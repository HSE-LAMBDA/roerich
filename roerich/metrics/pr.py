import numpy as np
import pandas as pd
from scipy.signal import argrelmax

def find_peaks(score, order):
    """
    Searches peaks in CPD score and estimates their heights.

    Parameters:
    -----------
    score: array-like
        Change point detection score.
    order: ind
        The number of points to the left and right of the peak to compare.

    Returns:
    --------
    cps_pred: numpy.array
        Indices of the detected change points.
    cps_score_pred: numpy.array
        Change point detection score for the detected indices of the change points.
    """

    cps_pred = argrelmax(score, order=order)[0]
    cps_score_pred = score[cps_pred]

    return cps_pred, cps_score_pred


def precision_recall_scores(cps_true, cps_pred, window=20):
    """
    Calculates precision and recall scores.

    Parameters:
    -----------
    cps_true: array-like
        True indices of change points. Example: [0, 100, 200, 300].
    cps_pred: array-like
        Detected indices of change points. Example: [0, 102, 195, 298, 303].
    window: int
        Maximum allowed distance between true and predicted change points.

    Retunrs:
    --------
    precision: float
        Precision score value.
    recall: float
        Recall score value.
    """

    cps_true = np.array(cps_true)
    cps_pred = np.array(cps_pred)

    n_cr = 0
    n_cp = len(cps_true)
    n_al = len(cps_pred)

    if n_al == 0:
        return 0, 0

    if n_cp == 0:
        return 0, 1

    is_used = []
    for atrue in cps_true:
        for apred in cps_pred:
            if (np.abs(apred - atrue) <= window) and (apred not in is_used):
                n_cr += 1
                is_used.append(apred)
                break

    tpr = n_cr / n_cp
    fpr = (n_al - n_cr) / n_al

    recall = tpr
    precision = 1 - fpr

    return precision, recall



def precision_score(cps_true, cps_pred, window=20):
    """
    Calculates precision score.

    Parameters:
    -----------
    cps_true: array-like
        True indices of change points. Example: [0, 100, 200, 300].
    cps_pred: array-like
        Detected indices of change points. Example: [0, 102, 195, 298, 303].
    window: int
        Maximum allowed distance between true and predicted change points.

    Retunrs:
    --------
    precision: float
        Precision score value.
    """

    precision, recall = precision_recall_scores(cps_true, cps_pred, window=20)
    return precision



def recall_score(cps_true, cps_pred, window=20):
    """
    Calculates recall score.

    Parameters:
    -----------
    cps_true: array-like
        True indices of change points. Example: [0, 100, 200, 300].
    cps_pred: array-like
        Detected indices of change points. Example: [0, 102, 195, 298, 303].
    window: int
        Maximum allowed distance between true and predicted change points.

    Retunrs:
    --------
    recall: float
        Recall score value.
    """

    precision, recall = precision_recall_scores(cps_true, cps_pred, window=20)
    return recall



def true_positive_rate(cps_true, cps_pred, window=20):
    """
    Calculates true positive rate.

    Parameters:
    -----------
    cps_true: array-like
        True indices of change points. Example: [0, 100, 200, 300].
    cps_pred: array-like
        Detected indices of change points. Example: [0, 102, 195, 298, 303].
    window: int
        Maximum allowed distance between true and predicted change points.

    Retunrs:
    --------
    tpr: float
        True positive rate value.
    """

    precision, recall = precision_recall_scores(cps_true, cps_pred, window=20)
    return recall



def false_positive_rate(cps_true, cps_pred, window=20):
    """
    Calculates false positive rate.

    Parameters:
    -----------
    cps_true: array-like
        True indices of change points. Example: [0, 100, 200, 300].
    cps_pred: array-like
        Detected indices of change points. Example: [0, 102, 195, 298, 303].
    window: int
        Maximum allowed distance between true and predicted change points.

    Retunrs:
    --------
    fpr: float
        False positive rate value.
    """

    precision, recall = precision_recall_scores(cps_true, cps_pred, window=20)
    return 1 - precision



def precision_recall_curve(cps_true, cps_pred, cps_score_pred=None, window=20):
    """
    Calculates precision-recall curve.

    Parameters:
    -----------
    cps_true: array-like
        True indices of change points. Example: [0, 100, 200, 300].
    cps_pred: array-like
        Detected indices of change points. Example: [0, 102, 195, 298, 303].
    cps_score_pred: array-like
        Change point detection score for the detected indices of the change points.
        If 'None', then the score for all change points is taken equal to 1.
    window: int
        Maximum allowed distance between true and predicted change points.

    Retunrs:
    --------
    thresholds: array-like
        Different thresholds for CPD scores of the detected change points.
    precision: array-like
        Precision score values according to the thresholds.
    recall: array-like
        Recall score values according to the thresholds.
    """

    if cps_score_pred is None:
        cps_score_pred = np.ones(cps_pred.shape)

    cps_true = np.array(cps_true)
    cps_pred = np.array(cps_pred)
    cps_score_pred = np.array(cps_score_pred)

    thresholds = np.unique(cps_score_pred)

    data = []
    for thr in thresholds:

        cps_thr = cps_pred[cps_score_pred >= thr]
        precision, recall = precision_recall_scores(cps_true, cps_thr, window)
        data.append([thr, precision, recall])

    data.insert(0, [-999, 0.0, 1.0])
    data.append([999, 1.0, 0.0])
    data = np.array(data)

    return data[:, 0], data[:, 1], data[:, 2]



def auc_score(thresholds, precision, recall):
    """
    Calculates auc.

    Parameters:
    -----------
    thresholds: array-like
        Different thresholds for CPD scores of the detected change points.
    precision: array-like
        Precision score values according to the thresholds.
    recall: array-like
        Recall score values according to the thresholds.

    Returns:
    --------
    auc: float
        The precision-recall curve auc value.
    """

    thresholds = np.array(thresholds)
    precision = np.array(precision)
    recall = np.array(recall)

    sorted_inds = thresholds.argsort()
    auc = np.abs(np.trapz(precision[sorted_inds], recall[sorted_inds]))

    return auc



def pr_auc(cps_true, cps_pred, cps_score_pred=None, window=20):
    """
    Calculates precision-recall curve auc.

    Parameters:
    -----------
    cps_true: array-like
        True indices of change points. Example: [0, 100, 200, 300].
    cps_pred: array-like
        Detected indices of change points. Example: [0, 102, 195, 298, 303].
    cps_score_pred: array-like
        Change point detection score for the detected indices of the change points.
        If 'None', then the score for all change points is taken equal to 1.
    window: int
        Maximum allowed distance between true and predicted change points.

    Retunrs:
    --------
    auc: float
        The precision-recall curve auc value.
    """

    thresholds, precision, recall = precision_recall_curve(cps_true, cps_pred, cps_score_pred, window)
    auc = auc_score(thresholds, precision, recall)

    return auc
