import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from .cpdc import ChangePointDetectionClassifier

# CV Classification
class ChangePointDetectionClassifierCV(ChangePointDetectionClassifier):
    
    def __init__(self, base_classifier='mlp', metric="klsym",
                 periods=1, window_size=100, step=1, n_runs=1, n_splits=5):
        """

        Change point detection algorithm based on binary classification [1] with Stratified KFold cross validation. 
        It takes to sliding windows (reference and test) in a signal, and separate them using the classifier. 
        The classification quality is considered as a change point detection score.

        [1] Mikhail Hushchyn and Andrey Ustyuzhanin. “Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation.” J. Comput. Sci. 53 (2021): 101385.

        Parameters:
        -----------
        base_classifier: {'logreg', 'qda', 'dt', 'rf', 'mlp', 'knn', 'nb'} or callable, default='mlp'
            Sklearn-like binary classifier to separate reference and test sliding windows in the signal.

            - 'logreg', Logistic Regression classifier,
              sklearn.linear_model.LogisticRegression()

            - 'qda', Quadratic Discriminant Analysis classifier,
              sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=0.01)

            - 'dt', Decision Tree classifier,
              sklearn.tree.DecisionTreeClassifier(min_samples_leaf=10, max_depth=6)

            - 'rf', Random Forest classifier,
              sklearn.ensemble.RandomForestClassifier((n_estimators=100, min_samples_leaf=10))

            - 'mlp', Multilayer Perceptron classifier,
              sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,100), solver="adam", activation="relu", learning_rate_init=0.1, max_iter=50, alpha=1.)

            - 'knn', K Neighbors classifier,
              sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)

            - 'nb', Gaussian Naive Bayes classifier,
              sklearn.naive_bayes.GaussianNB(var_smoothing=0.01)

            - Callable sklearn-like classifier,
              Example: base_classifier = LogisticRegression()

        metric: {'klsym', 'pesym', 'jsd', 'mmd', 'fd'} or callable, default='klsym'.
            {'KL', 'PE'} will be deprecated in future versions.
            Name of a score function, that is used to measure the classifier quality based on predictions
            for reference (p_ref) and test (p_test) windows. It is considered as change point detection score.

            - 'klsym' or 'KL_sym', symmetric Kullback-Leibler (KL) divergence,
            KL(p_test||p_ref) + KL(p_ref||p_test)

            - 'pesym' or 'PE_sym', symmetric Pearson (PE) divergence,
            PE(p_test||p_ref) + PE(p_ref||p_test)

            - 'jsd' or 'JSD', Jensen–Shannon divergence (JSD),
            JSD(p_test||p_ref)

            - 'mmd', the Maximum Mean Discrepancy (MMD),
            MMD(p_test, p_ref)

            - 'fd', the Frechet Distance (FD),
            FD(p_test, p_ref)

            - Callable function,
            Example: metric = roerich.scores.frechet_distance

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
        
        n_splits: int, default=5
            Number of splits for KFold cross validation.

        """

        super().__init__(base_classifier=base_classifier, metric=metric, periods=periods,
                         window_size=window_size, step=step, n_runs=n_runs)
        self.n_splits = n_splits


    @ignore_warnings(category=ConvergenceWarning)
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

        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=np.random.randint(0, 1000))
        y_pred_glob = []
        y_test_glob = []

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test, y_train, y_test = (X[train_index], 
                                                X[test_index], 
                                                y[train_index], 
                                                y[test_index])

            ss = StandardScaler()
            ss.fit(X_train)
            X_train = ss.transform(X_train)
            X_test = ss.transform(X_test)

            classifier = deepcopy(self.base_classifier)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict_proba(X_test)[:, 1]
            y_pred_glob.append(y_pred)
            y_test_glob.append(y_test)

        y_pred = np.concatenate(tuple(y_pred_glob), axis=0)
        y_test = np.concatenate(tuple(y_test_glob), axis=0)
        score = self.metric(y_pred[y_test == 0], y_pred[y_test == 1])

        return score