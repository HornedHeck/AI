from torch import nn

from general import MNIST_DATA_SIZE


def get_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(MNIST_DATA_SIZE, 500),
        nn.LeakyReLU(),
        nn.Linear(500, 100),
        nn.LeakyReLU(),
        nn.BatchNorm1d(100),
        nn.Linear(100, 25),
        nn.LeakyReLU(),
        nn.Linear(25, 10)
    )
