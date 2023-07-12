import numpy as np
from sklearn.preprocessing import StandardScaler



def generate_dataset(period=200, N_tot=1000):
    mu = 0
    sigma = 1.
    N = 1

    T = [0, 1]
    X = [np.random.normal(mu, sigma, 1)[0], np.random.normal(mu, sigma, 1)[0]]
    cps = []

    for i in range(2, N_tot):
        if i % period == 0:
            N += 1
            mu += 0.5 * N
            cps.append(i)
        T += [i]
        ax = 0.6 * X[i-1] - 0.5 * X[i-2] + np.random.normal(mu, sigma, 1)[0]
        X += [ax]
    return np.array(X).reshape(-1, 1), cps


def generate_normal(periods=200, n_changes=5, n_features=1, seed=None):

    X = []
    cps = []

    mu = 0
    sigma_log = 0
    np.random.seed(seed)
    mu_sign, sigma_sign = np.random.choice([-1, 1], 2)

    for i in range(n_changes):
        k = np.random.randint(0, 3)
        mu_jump = np.random.uniform(1, 5, (n_features, ))
        log_jump = np.log(np.random.uniform(2, 4, (n_features, )))
        if k == 0:
            mu += mu_sign * mu_jump
            mu_sign *= -1
        if k == 1:
            sigma_log += sigma_sign * log_jump
            sigma_sign *= -1
        if k == 2:
            mu += mu_sign * mu_jump
            mu_sign *= -1
            sigma_log += sigma_sign * log_jump
            sigma_sign *= -1

        ax = np.random.normal(mu, np.exp(sigma_log), (periods, n_features))
        X.append(ax)
        cps += [periods * (i+1)]

    X = np.concatenate(tuple(X), axis=0)
    X = StandardScaler().fit_transform(X)

    return X, cps[:-1]
