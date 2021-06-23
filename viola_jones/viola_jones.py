from matplotlib.image import imread
import numpy as nmp
from visualizer import show_faced, show_image
from dataset_utils import get_data


def to_integral(src: nmp.ndarray) -> nmp.ndarray:
    res = nmp.zeros(src.shape)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = src[i, j] - res[i, j] + res[i, j - 1] + res[i - 1, j]
    return res








if __name__ == '__main__':
    x, y = get_data()
