import numpy as np
import pandas as pd

from scipy import interpolate


def unified_score(T: np.ndarray, T_score: np.ndarray, score: np.ndarray) -> np.ndarray:
    """
    Interpolates a CPD score of T_score interval onto T interval
    Parameters
    ----------
    T: A broader time-step interval
    T_score: A time intervals of CPD scores
    score: A CPD scores

    Returns
    -------
    Interpolated CPD scores
    """
    inter = interpolate.interp1d(T_score, score, kind='previous', fill_value=(0, 0), bounds_error=False)
    uni_score = inter(T)
    return uni_score


def autoregression_matrix(X, periods=1, fill_value=0):
    shifted_x = [pd.DataFrame(X).shift(periods=i, fill_value=fill_value).values for i in range(periods)]
    X_auto = np.hstack(tuple(shifted_x))
    return X_auto


def reference_test(X, window_size=2, step=1):
    T = []
    reference = []
    test = []
    for i in range(2 * window_size - 1, len(X), step):
        T.append(i)
        reference.append(X[i - 2 * window_size + 1:i - window_size + 1])
        test.append(X[i - window_size + 1:i + 1])
    return np.array(T), np.array(reference), np.array(test)
