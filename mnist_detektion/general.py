MNIST_DATA_SIZE: int = 28 * 28
BATCH_SIZE = 20
EPOCH_SIZE = 200
EPOCH_BATCH_COUNT = EPOCH_SIZE // BATCH_SIZE
TRAIN_EPOCH_COUNT = 100

from scipy.interpolate import make_interp_spline
import numpy as nmp
import matplotlib.pyplot as plt


def interpolate_plot(y_src, size: int, name: str):
    x_src = nmp.arange(size)
    y = make_interp_spline(x_src, y_src, k=3)
    x = nmp.linspace(0, size - 1, 100)
    plt.plot(x, y(x), label=name)
    # plt.plot(x_src, y_src)
