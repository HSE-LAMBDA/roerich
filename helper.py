import numpy as np 
from matplotlib import pyplot as plt


def dataset1(period=200, N_tot=1000):
    mu = 0
    sigma = 1.
    N = 1
    
    T = [0, 1]
    X = [np.random.normal(mu, sigma, 1)[0], np.random.normal(mu, sigma, 1)[0]]
    
    for i in range(2, N_tot):
        if i % period == 0:
            N += 1
            mu += 0.5 * N
        T += [i]
        ax = 0.6 * X[i-1] - 0.5 * X[i-2] + np.random.normal(mu, sigma, 1)[0]
        X += [ax]
    return np.array(X).reshape(-1, 1)


def plot(X, Y, title, x_label, y_label):
    plt.figure(figsize=(12, 3.))
    plt.plot(X, Y, linewidth=3)

    plt.xlabel(x_label, size=14)
    plt.ylabel(y_label, size=14)
    plt.grid(b=1)
    plt.title(title, size=14)
    # plt.legend(loc='best')
    plt.tight_layout()
    plt.xlim(0)
    #plt.show()


def plot2(title, x_label, y_label):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(b=1)
    #ax.plot([1, 2, 3], [2, 4, 6])
    ax.set_title(title)
    plt.tight_layout()
    # ax.set_xlim(0)
    return ax

def draw_projection(ax, score, period, ws, n):
    min_height, max_height = np.min(score), np.max(score)
    for line in range(1, n):
        current_period = period * line
        c_ws = current_period + ws
        ax.plot([current_period, current_period], [min_height, max_height], '--', alpha=1, lw=1.5, c='k')
        ax.plot([c_ws, c_ws], [min_height, max_height], '-.', alpha=0.5, lw=1, c='k')
    return ax
    
def SMA(scores, N):
    new_scores = []
    for i in range(0, len(scores)):
        s = i - N if i - N >= 0 else 0
        new_scores.append(np.mean(scores[s:i+1], axis=0))
    return np.array(new_scores)


def EMA(scores, N, smooth):
    alpha = smooth / (1. + N)
    new_scores = [scores[0]]
    for i in range(1, len(scores)):
        new_score = alpha * scores[i] + (1 - alpha) * new_scores[i-1]
        new_scores.append(new_score)
    return new_scores


def cum_sum(scores):
    new_scores = [0.]
    for i in range(1, len(scores)):
        new_scores.append(np.maximum(0., new_scores[i-1] + scores[i]))
    return new_scores


def generate_simplest_data(period=200, N=2):
    X = []
    for i in range(N):
        temp = [i for j in range(period)]
        X += temp
    return np.array(X).reshape(-1, 1)


def generate_normal_simple_data(period=200, N_tot=1000):
    mu = 0
    sigma = 0.1
    N = 1
    
    T = [0, 1]
    X = [np.random.normal(mu, sigma, 1)[0], np.random.normal(mu, sigma, 1)[0]]
    
    for i in range(2, N_tot):
        if i % period == 0:
            N += 1
            mu += 0.5 * N
        T += [i]
        ax = 0.6 * X[i-1] - 0.5 * X[i-2] + np.random.normal(mu, sigma, 1)[0]
        X += [ax]
    return np.array(X).reshape(-1, 1)