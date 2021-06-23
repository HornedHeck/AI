import matplotlib.pyplot as plt
import numpy as nmp
from matplotlib.patches import Rectangle


def show_faced(image: nmp.ndarray, rect: iter, real_rect: iter = None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = Rectangle(rect[:2], rect[2] - rect[0], rect[3] - rect[1], edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if real_rect is not None:
        rect = Rectangle(real_rect[:2], real_rect[2] - real_rect[0], real_rect[3] - real_rect[1], edgecolor='b',
                         facecolor='none')
        ax.add_patch(rect)
    plt.show()


def show_image(data: nmp.ndarray):
    plt.imshow(data)
    plt.show()
