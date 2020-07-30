import numpy as np
from matplotlib import pyplot as plt


def display(X, T, L, S, Ts, peaks=None, plot_peak_height=10, s_max=10):
    n = X.shape[1] + 1 if peaks is None else X.shape[1] + 2
    
    plt.figure(figsize=(12, n*2.5+0.25))
    
    for i in range(X.shape[1]):
        
        plt.subplot(n, 1, i+1)
        ax = X[:, i]
        plt.plot(T, ax, linewidth=2, label='Original signal', color='C0')
        for t in T[L == 1]:
            plt.plot([t]*2, [ax.min(), ax.max()], color='0', linestyle='--')
        plt.ylim(ax.min(), ax.max())
        plt.xlim(0, T.max())
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.legend(loc='upper left', fontsize=16)
        plt.tight_layout()
    
    score_plot_ix = n if peaks is None else n - 1
    plt.subplot(n, 1, score_plot_ix)
    plt.plot(Ts, S, linewidth=3, label="Change-point score", color='C3')
    for t in T[L == 1]:
        plt.plot([t]*2, [-1, s_max], color='0', linestyle='--')
    
    # display find peaks #todo refactoring
    if peaks is not None:
        plt.subplot(n, 1, n)
        new_score_peaks = np.zeros(len(T))
        new_score_peaks[peaks] = plot_peak_height
        plt.plot(new_score_peaks, linewidth=3, label="Peaks", color='C4')
        for t in T[L == 1]:
            plt.plot([t]*2, [-1, s_max], color='0', linestyle='--')
    
    plt.ylim(-1, s_max)
    plt.xlim(0, T.max())
    plt.xticks(size=16)
    plt.yticks(np.arange(0, s_max+1, 5), size=16)
    plt.xlabel("Time", size=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.tight_layout()