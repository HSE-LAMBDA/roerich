import numpy as np
from scipy.signal import find_peaks
from densratio import densratio
from joblib import Parallel, delayed

from .utils import autoregression_matrix, unified_score, reference_test


class ChangePointDetectionRuLSIF(object):
    
    def __init__(self, alpha=0.1, kernel_num=100,
                 periods=1, window_size=100, step=1, n_runs=1, debug=0):
        self.alpha = alpha
        self.kernel_num = kernel_num
        self.periods = periods
        self.window_size = window_size
        self.step = step
        self.n_runs = n_runs
        self.debug = debug
    
    def densration_gridsearch(self, X_ref, X_test):
        lambda_range = 10 ** np.linspace(-3, 3, 7)  # np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1])
        sigma_range = 10 ** np.linspace(-3, 3,
                                        25)  # np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3])
        estimator_1 = densratio(X_ref, X_test, self.alpha, sigma_range, lambda_range, self.kernel_num, verbose=False)
        estimator_2 = densratio(X_test, X_ref, self.alpha, sigma_range, lambda_range, self.kernel_num, verbose=False)
        w1_ref = estimator_1.compute_density_ratio(X_ref)
        w2_test = estimator_2.compute_density_ratio(X_test)
        score_max = (0.5 * np.mean(w1_ref) - 0.5) + (0.5 * np.mean(w2_test) - 0.5)
        return score_max
    
    def reference_test_predict(self, X_ref, X_test):
        score = self.densration_gridsearch(X_ref, X_test)
        return score
    
    def reference_test_predict_n_times(self, X_ref, X_test):
        scores = []
        for i in range(self.n_runs):
            ascore = self.reference_test_predict(X_ref, X_test)
            scores.append(ascore)
        return np.mean(scores)
    
    def predict(self, X, distance=5, height=None, smooth=False):
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)
        scores = []
        T_scores = []
        iters = range(0, len(reference), self.step)
        scores = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_scores = np.array([T[i] for i in iters])
        
        T = np.arange(len(X))
        scores = unified_score(T, T_scores - self.step, scores)
        
        shift = self.window_size
        scores = unified_score(T, T - shift, scores)
        
        if smooth:
            from scipy.signal import savgol_filter
            width = int((np.round(0.25 * self.window_size) // 2) * 2 + 1)
            scores = savgol_filter(scores, width, 1)
        
        width = 0.25 * (self.window_size)
        peaks, _ = find_peaks(scores, distance=distance, width=width, height=height)
        
        return np.array(scores), peaks
