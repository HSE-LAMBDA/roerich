from ..algorithms.cpdc import ChangePointDetectionBase
from roerich.scores.fd import frechet_distance
from roerich.scores.mmd import maximum_mean_discrepancy
from roerich.scores.energy import energy_distance
import numpy as np
from sklearn.utils import resample


class SlidingWindows(ChangePointDetectionBase):

    def __init__(self, metric='mmd', bootstrap=False, periods=1, window_size=100, step=1, n_runs=1):
        super().__init__(periods=periods, window_size=window_size, step=step, n_runs=n_runs)
        self.metric = metric
        self.bootstrap = bootstrap

        """
        Sliding windows change point detection method

        Parameters:
        -----------
        
        metric: str/function, default=None
            Possible values: {'energy', 'fd', 'mmd'}.
        'energy' - energy distance
        'fd' - Frechet distance
        'mmd' - maximum mean discrepancy
        Otherwise, a function of format like in frechet_distance from roerich.scores.fd
        
        bootstrap: bool, default=False
            Whether to use bootstrapping procedure. Resampling is done n_runs times and the average score is returned
        
        periods: int, default=1
            Number of consecutive observations of a time series, considered as one input vector.
        The signal is considered as an autoregression process (AR) for classification. In the most cases periods=1
        will be a good choice.

        window_size: int, default=100
            Number of consecutive observations of a time series in test and reference
        windows. Recommendation: select the value so that there is only one change point within 2*window_size
        observations of the signal.

        step: int, default=1
            Algorithm estimates change point detection score for each <step> observation. step > 1 helps
        to speed up the algorithm.

        n_runs: int, default=1
            Number of times the bootstrapping is applied
            
        """

    def reference_test_predict(self, X_ref, X_test):

        if self.metric == "energy":
            return energy_distance(X_ref, X_test)
        elif self.metric == "fd":
            return frechet_distance(X_ref, X_test)
        elif self.metric == "mmd":
            return maximum_mean_discrepancy(X_ref, X_test)
        elif callable(self.metric):
            return self.metric(X_ref, X_test)
        else:
            raise ValueError("metric should be one of: energy, fd, mmd; or a function should be "
                             "passed")

    def reference_test_predict_n_times(self, X_ref, X_test):
        """
        Estimate change point detection score several times for a pair of test and reference windows
        where bootstrapping procedure is applied to the test window

        Parameters:
        -----------
        X_ref: numpy.ndarray
            Matrix of reference observations.
        X_test: numpy.ndarray
            Matrix of test observations.

        Returns:
        --------
        score: float
            Estimated average change point detection score for a pair of window.
        """

        scores = []
        for i in range(self.n_runs):
            if self.bootstrap:
                X_ref_b, X_test_b = resample(X_ref), resample(X_test)
            else:
                X_ref_b, X_test_b = X_ref, X_test
            ascore = self.reference_test_predict(X_ref_b, X_test_b)
            scores.append(ascore)

        return np.mean(scores)
