from .cpdc import *
from sklearn.metrics import pairwise_distances


class EnergyDistanceCalculator(ChangePointDetectionBase):

    def __init__(self, window_size=100,  periods=1, step=1, n_runs=1):
        """
        Change point detection algorithm based on energy distance.

        Parameters:
        -----------
        window_size: int
            Number of consecutive observations of a time series in test and reference windows.
        step: int
            Algorithm estimates change point detection score for each <step> observation.
        n_runs: int
            Number of times, the binary classifier runs on each pair of test and reference windows.
        """

        self.window_size = window_size
        self.periods = periods
        self.step = step
        self.n_runs = n_runs



    def reference_test_predict(self, X_ref, X_test):
        """
        Estimate change point detection score for a pair of test and reference windows.

        Parameters:
        -----------
        X_ref: numpy.ndarray
            Matrix of reference observations.
        X_test: numpy.ndarray
            Matrix of test observations.

        Retunrs:
        --------
        score: float
            Estimated change point detection score for a pair of window.
        """

        n = X_ref.shape[0]
        E = 2*pairwise_distances(X_ref, X_test, metric='euclidean') - pairwise_distances(X_test, metric='euclidean') - pairwise_distances(X_ref, metric='euclidean')
        return np.sum(E)/n**2


    def reference_test_predict_n_times(self, X_ref, X_test):
        """
        Estimate change point detection score several times for  a pair of test and reference windows.

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
            ascore = self.reference_test_predict(X_ref, X_test)
            scores.append(ascore)
        
        return np.mean(scores)


    def posprocessing(self, T, T_score, score):
        """
        Interpolates and shifts a change point detection score, estimates peak positions.
        
        Parameters:
        -----------
        T: numpy.array
            A broader time-step interval
        T_score: numpy.array
            A time intervals of CPD scores
        score: numpy.array
            A CPD scores

        Returns:
        --------
        new_score: numpy.array
            Interpolated and shifted CPD scores.
        peaks: numpy.array
            Positions of peaks in the CPD scores.
        """

        inter = interpolate.interp1d(T_score-self.window_size, score, 
                                     kind='linear', fill_value=(0, 0), bounds_error=False)
        new_score = inter(T)
        peaks = argrelmax(new_score, order=self.window_size)[0]

        return new_score, peaks


    def predict(self, X):
        """
        Estimate change point detection score for a time series.

        Parameters:
        -----------
        X: numpy.ndarray
            Time series observation.

        Retunrs:
        --------
        T_score: numpy.array
            Array of timestamps.
        scores: numpy.array
            Estimated change point detection score.
        """
        
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)
        
        scores = []
        T_score = []
        iters = range(0, len(reference), self.step)
        score = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_score = np.array([T[i] for i in iters])

        new_score, peaks = self.posprocessing(np.arange(len(X)), T_score, score)
        
        return new_score, peaks

