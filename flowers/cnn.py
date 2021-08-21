import torch
from torch import Tensor
from torch.nn import Sequential, Conv2d, ReLU, NLLLoss, MaxPool2d, LogSoftmax, Linear, Dropout, Flatten, BatchNorm1d

from base.BaseModel import BaseModel


class FlowersCNN(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = Sequential(

            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Flatten(),
            Linear(25088, 8192),
            Dropout(),
            BatchNorm1d(8192),
            Linear(8192, 1024),
            Dropout(),
            BatchNorm1d(1024),
            Linear(1024, 102),
            LogSoftmax(dim=1)
        )
        self.__loss__ = NLLLoss()

    def forward(self, x):
        return self.model(x.view((-1, 3, 500, 500)).float())

    @property
    def name(self) -> str:
        return 'flowers_cnn'

    def loss(self, r: Tensor, y: Tensor) -> Tensor:
        return self.__loss__(r, y)

    def is_correct(self, r: Tensor, y: Tensor) -> int:
        return (r.argmax(1) == y).sum()

    def clear(self):
        super().clear()

    def clear_grad(self):
        super().clear_grad()


if __name__ == '__main__':
    x = torch.zeros((10, 3, 500, 500))
    model = FlowersCNN()
    res = model(x)
    print(res.shape)
