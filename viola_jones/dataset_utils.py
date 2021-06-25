import os
import random

import cv2
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


def create_info():
    with open(f'faces.info', 'w+') as info:
        for f in os.listdir(faces_path):
            img = cv2.imread(f'{faces_path}{f}', cv2.IMREAD_UNCHANGED)
            info.write(f'{faces_path}{f} 1 0 0 {img.shape[0]} {img.shape[1]}\n')
    with open(f'empty.info', 'w+') as info:
        for f in os.listdir(empty_path):
            info.write(f'{empty_path}{f}\n')


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


def add_faces():
    downloads_path = '/home/hornedheck/Загрузки/faced/'
    cls = cv2.CascadeClassifier('/home/hornedheck/PycharmProjects/AI/models/anime_v2.xml')
    info = []
    counter = len(os.listdir(f'{path}d_faces/')) + 1
    for file in os.listdir(downloads_path):
        img = cv2.imread(f'{downloads_path}{file}', cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cls.detectMultiScale(gray, 1.1, 1)
        for (x, y, w, h) in faces:
            face = img[y:y + h + 1, x:x + w + 1]
            info.append(f'd_faces/face_{counter}.png')
            cv2.imwrite(f'{path}d_faces/face_{counter}.png', face)
            counter += 1

    for i in info:
        print(i)


if __name__ == '__main__':
    create_info()
