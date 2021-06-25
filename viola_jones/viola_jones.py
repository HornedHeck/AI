import random
import time

from matplotlib.image import imread
import numpy as nmp
from visualizer import show_rects, show_image
from classifier_ada import exclusive_argmin
from dataset_utils import get_data, to_integral
from features import generate_horizontal_rects, generate_vertical_3rects, generate_test, Feature

feature_params = {
    'r_eye': [130, 45, 165, 130, 15, 60, 1, 0.2],
    'l_eye': [35, 45, 75, 130, 15, 60, 1, 0.2],
    'nose': [60, 80, 130, 120, 50, 25, 0.5, 0.15],
    'mouth_1': [50, 80, 150, 150, 50, 40, 0.75, 0.4],
    'mouth_2': [60, 110, 130, 150, 50, 30, 0.5, 0.2],
    'face_1': [50, 50, 150, 120, 70, 40, 1, 0.25],
    'face_2': [75, 25, 125, 150, 35, 100, 1.5, 0.25]
}


def q(feature: Feature, x: nmp.ndarray, y: nmp.ndarray, weights: nmp.ndarray):
    res = 0.
    for i in range(y.shape[0]):
        if feature.calc_feature(x[i]) != y[i]:
            res += weights[i]

    if res > 0.5:
        feature.s = - feature.s
        res = 1 - res
    return res


def q_v(features: iter, x: nmp.ndarray, y: nmp.ndarray, weights: nmp.ndarray):
    return nmp.array([q(feature, x, y, weights) for feature in features])


def feature_exps(a: float, y: nmp.ndarray, x: nmp.ndarray, feature: Feature):
    return nmp.exp(nmp.array([-a * y[i] * feature.calc_feature(x[i]) for i in range(len(y))]))


def learn(features: iter, iterations: int):
    x, y = get_data()
    a = nmp.zeros(len(features))
    selected = []
    x = nmp.vectorize(to_integral, signature='(n,m)->(n,m)')(x)
    weights: nmp.ndarray = nmp.ones(y.shape) / y.shape[0]
    for i in range(iterations):
        errors = q_v(features, x, y, weights)
        t = exclusive_argmin(errors, selected)
        selected.append(t)
        a[t] = 0.5 * nmp.log((1 - errors[t]) / errors[t])
        weights = weights * feature_exps(a[t], y, x, features[t])
        weights = weights / weights.sum()
        print(f'Iteration {i + 1}: {t}, {errors[t]}')

    for i, a_i in enumerate(a):
        print(f'{a_i},', end='')
        # if a_i > 0.1 or a_i < -0.1:
        # print(f'{i}, {a_i}, {features[i]}')


if __name__ == '__main__':
    print('mouth_1, mouth_2')
    features = generate_vertical_3rects(*feature_params['mouth_2'])
    features += generate_horizontal_rects(*feature_params['mouth_1'])
    print(len(features))
    learn(
        features,
        10
    )
    # x, _ = get_data()
    # f = generate_test(*feature_params['r-eye'], 4)[3]
    # show_rects(x[7], [[f.lx, f.ly, f.rx, f.ry]])
