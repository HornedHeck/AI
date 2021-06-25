import matplotlib.pyplot as plt
import numpy as nmp
from matplotlib.patches import Rectangle


def show_rects(image: nmp.ndarray, rects: iter):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    colors = plt.cm.get_cmap('hsv', len(rects))
    for i in range(len(rects)):
        rect = Rectangle(
            rects[i][:2],
            rects[i][2] - rects[i][0],
            rects[i][3] - rects[i][1],
            facecolor='none',
            edgecolor=colors(i)
        )
        ax.add_patch(rect)
    plt.show()


def show_image(data: nmp.ndarray):
    plt.imshow(data, cmap='gray')
    plt.show()


def show_prediction(image: nmp.ndarray, expected: int, got):
    plt.title(f'Expected {expected} , got {got}')
    show_image(image)
