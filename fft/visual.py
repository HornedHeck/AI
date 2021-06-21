import matplotlib.pyplot as plt
import numpy as nmp


def visualize_track(data: nmp.ndarray):
    # data = data[200000:2200000]
    x = nmp.arange(0, data.shape[0])
    plt.plot(x, data[:, 0], label='L')
    plt.plot(x, data[:, 1], label='R')
    plt.legend()
    plt.show()


def plot_spectrum(data: nmp.ndarray):
    plt.plot(data)
    plt.show()


def show_spectrum_map(spectrum_map: nmp.ndarray):
    plt.imshow(spectrum_map)
    plt.show()
