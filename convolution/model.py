import torch
from torch import nn


def get_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(3136, 1024),
        nn.Dropout(),
        nn.Linear(1024, 10),
        nn.LogSoftmax()
    )


if __name__ == '__main__':
    size = 28
    batches = 10
    test = torch.rand([batches, size, size])
    test = torch.reshape(test, [batches, 1, size, size])
    model = get_model()
    res = model(test).detach()
    print(res.shape)
