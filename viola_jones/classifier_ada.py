import numpy as nmp

from dataset_utils import get_data, to_integral
from features import Feature


class Classifier(object):

    def __init__(self, a: nmp.ndarray, features: list[Feature]) -> None:
        super().__init__()
        self.a = a
        self.features = features

    def calc(self, image: nmp.ndarray) -> int:
        res = 0.
        for i in range(len(self.a)):
            res += self.a[i] * self.features[i].calc_feature(image)
        return int(nmp.sign(res))

    def __str__(self) -> str:
        items = len(self.a)
        res = '['
        for i in range(items):
            res += f'{self.a[i]}: {self.features[i]}'
            if i < items - 1:
                res += ','
        return res + ']'


def q(classifier: Classifier, x: nmp.ndarray, y: nmp.ndarray, weights: nmp.ndarray):
    res = 0.
    for i in range(y.shape[0]):
        if classifier.calc(x[i]) != y[i]:
            res += weights[i]

    return res


def q_v(features: iter, x: nmp.ndarray, y: nmp.ndarray, weights: nmp.ndarray):
    return nmp.array([q(feature, x, y, weights) for feature in features])


def classifier_exps(a: float, y: nmp.ndarray, x: nmp.ndarray, classifier: Classifier):
    return nmp.exp(nmp.array([-a * y[i] * classifier.calc(x[i]) for i in range(len(y))]))


def exclusive_argmin(data: nmp.ndarray, excluded: list[int]) -> int:
    min_i = -1

    for i in range(data.shape[0]):
        if min_i == -1 and i not in excluded:
            min_i = i
        elif i not in excluded and data[i] < data[min_i]:
            min_i = i
    return min_i


def learn(classifiers: list[Classifier], iterations: int):
    x, y = get_data()
    selected = []
    a = nmp.zeros(len(classifiers))
    x = nmp.vectorize(to_integral, signature='(n,m)->(n,m)')(x)
    weights: nmp.ndarray = nmp.ones(y.shape) / y.shape[0]
    for i in range(iterations):
        errors = q_v(classifiers, x, y, weights)
        t = exclusive_argmin(errors, selected)
        selected.append(t)
        a[t] = 0.5 * nmp.log((1 - errors[t]) / errors[t])
        weights = weights * classifier_exps(a[t], y, x, classifiers[t])
        weights = weights / weights.sum()
        print(f'Iteration {i + 1}: {t}, {errors[t]}')

    for i, a_i in enumerate(a):
        print(f'{i}, {a_i}, {classifiers[i]}')
