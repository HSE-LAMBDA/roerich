from .cpdc import ChangePointDetectionBase
from roerich.scores.fd import frechet_distance
from roerich.scores.mmd import maximum_mean_discrepancy
from roerich.scores.energy import energy_distance


class SlidingWindows(ChangePointDetectionBase):

    def __init__(self, metric=None, periods=1, window_size=100, step=1, n_runs=1):
        super().__init__(periods=periods, window_size=window_size, step=step, n_runs=n_runs)
        self.metric = metric

        """
        Change point detection algorithm based on binary classification.

        Parameters:
        -----------
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
            Number of times, the binary classifier runs on each pair of test and reference
        windows. Observations in the windows are divided randomly between train and validation sample every time.
        n_runs > 1 helps to reduce noise in the change point detection score.
        
        metric: str/function, default=None
            Function that gives the measure of dissimilarity between data points in windows.
        Metric should be one of: EnergyDist, FrechetDist, MaxMeanDisc; or a function should be passed.
        Function must be in the following format:

            Parameters:
            -----------
            X_ref: numpy.ndarray
                Matrix of reference observations.
            X_test: numpy.ndarray
                Matrix of test observations.
    
            Returns:
            --------
            score: float
                Estimated change point detection score for a pair of window.

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
            raise ValueError("metric should be one of: EnergyDist, FrechetDist, MaxMeanDisc; or a function should be "
                             "passed")
