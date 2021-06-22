from matplotlib.image import imread
import numpy as nmp
from visualizer import show_faced

if __name__ == '__main__':
    data = imread('../dataset/faces/vivy_02.jpg')
    show_faced(data, [100, 100, 200, 200])
