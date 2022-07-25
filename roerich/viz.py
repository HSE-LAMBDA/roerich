import numpy as np
from matplotlib import pyplot as plt


def display(X, cps_true, score=None, cps_pred=None, plot_peak_height=10, s_max=10):

    T = np.arange(len(X))
    L = np.zeros(len(X))
    L[cps_true] = 1

    n = X.shape[1] + 1 if cps_pred is None else X.shape[1] + 2

    plt.figure(figsize=(12, n*2.5+0.25))

    for i in range(X.shape[1]):

        ax = X[:, i]
        d = 0.05 * (ax.max() - ax.min())

        plt.subplot(n, 1, i+1)
        plt.plot(T, ax, linewidth=2, label='Original signal', color='C0')
        for t in T[L == 1]:
            plt.plot([t]*2, [ax.min()-d, ax.max()+d], color='0', linestyle='--')
        plt.ylim(ax.min()-d, ax.max()+d)
        plt.xlim(0, T.max())
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.legend(loc='upper left', fontsize=16)
        plt.tight_layout()

    score_plot_ix = n if cps_pred is None else n - 1
    if score is not None:
        d = 0.05 * (score.max() - score.min())
        plt.subplot(n, 1, score_plot_ix)
        plt.plot(T, score, linewidth=3, label="Change point score", color='C3')
        for t in T[L == 1]:
            plt.plot([t]*2, [score.min()-d, score.max()+d], color='0', linestyle='--')
        plt.ylim(score.min()-d, score.max()+d)
        plt.xlim(0, T.max())
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.legend(loc='upper left', fontsize=16)
        plt.tight_layout()

    # display find peaks #todo refactoring
    if cps_pred is not None:
        plt.subplot(n, 1, n)
        L_pred = np.zeros(len(T))
        L_pred[cps_pred] = 1
        plt.plot(L_pred, linewidth=3, label="Change points", color='C4')
        for t in T[L == 1]:
            plt.plot([t]*2, [-0.05, 1.05], color='0', linestyle='--')
        plt.ylim(-0.05, 1.05)
        plt.xlim(0, T.max())
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.legend(loc='upper left', fontsize=16)

    plt.xlabel("Time", size=16)
    plt.tight_layout()
