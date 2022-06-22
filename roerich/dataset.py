import numpy as np



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
