import numpy as np

from helper import plot2, draw_projection, SMA
from matplotlib import pyplot as plt 


def create_scaled_data_ref_test(scaler, ref, test):
    scaled_data = []
    for i in range(ref.shape[0]):
        new_x = np.vstack([ref[i], test[i]])
        scaler.fit(new_x)
        transformed_data = scaler.transform(ref[i])
        scaled_data.append(transformed_data)
    return np.array(scaled_data)

def create_scaled_data_ref(scaler, ref):
    scaled_data = []
    for i in range(ref.shape[0]):
        scaler.fit(ref[i])
        transformed_data = scaler.transform(ref[i])
        scaled_data.append(transformed_data)
    return np.array(scaled_data)

def plotter_scale(T, scaled_data, ws):
    ax = plot2('Scaled', 'X', 'Timestamp')
    x_data = scaled_data.mean(axis=1)
    x_data = x_data.mean(axis=1)
    ax.plot(T, SMA(x_data, 1), linewidth=2, alpha=1)
    
    draw_projection(ax, scaled_data, 2000, ws, 10)
    
    ax.legend(loc='best')
    plt.show()

    
def plotter_scale_many_x(T, scaled_datas, ws, labels, ylims):
    ax = plot2('Scaled', 'X', 'Timestamp')
    for i, scaled_data in enumerate(scaled_datas):
        x_data = scaled_data.mean(axis=1)
        x_data = x_data.mean(axis=1)
        x_data = SMA(x_data, 200)
        ax.plot(T, x_data, linewidth=2, alpha=1, label=labels[i])
    draw_projection(ax, scaled_data, 2000, ws, 10)
    ax.legend(loc='best')
    ax.set_ylim(*ylims)
    plt.show()