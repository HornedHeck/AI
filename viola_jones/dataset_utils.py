import os
import random

import matplotlib.pyplot as plt
import numpy as nmp

path = '/home/hornedheck/PycharmProjects/AI/dataset/Anime/'
faces_path = f'{path}faces/'
empty_path = f'{path}empty/'
image_size = 192


def grayscale(data: nmp.ndarray) -> nmp.ndarray:
    res: nmp.ndarray = data[:, :, 0] * 0.3 + data[:, :, 1] * 0.59 + data[:, :, 2] * 0.11
    res: nmp.ndarray = res / res.max()
    res = (res * 255).astype(int)
    return res


def to_integral(src: nmp.ndarray) -> nmp.ndarray:
    res = nmp.zeros(src.shape)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = src[i, j] + res[i - 1, j] + res[i, j - 1]
            if j > 0:
                res[i, j] -= res[i - 1, j - 1]
    return res


def get_photos(src_path: str) -> nmp.ndarray:
    return nmp.array([grayscale(plt.imread(src_path + photo)) for photo in os.listdir(src_path)])


def get_data():
    faces = get_photos(faces_path)
    empty = get_photos(empty_path)
    return nmp.vstack((faces, empty)), nmp.hstack(
        (nmp.ones(faces.shape[0], dtype=int), -nmp.ones(empty.shape[0], dtype=int)))


def showcase():
    x, y = get_data()
    for i in range(10):
        j = random.randint(0, y.shape[0] - 1)
        plt.title(f'{y[j]}')
        plt.imshow(x[j], cmap='gray')
        plt.show()


def add_empty():
    downloads_path = '/home/hornedheck/Загрузки/empty/'

    empty_counter = len(os.listdir(empty_path)) + 1

    for f in os.listdir(downloads_path):
        img: nmp.ndarray = plt.imread(f'{downloads_path}{f}')[:, :, :3]
        c_x = img.shape[0] // image_size
        c_y = img.shape[1] // image_size
        for i in range(c_x):
            for j in range(c_y):
                part = img[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size, ]
                plt.imsave(f'{empty_path}empty_{empty_counter:02d}.png', part)
                empty_counter += 1
        os.remove(f'{downloads_path}{f}')
