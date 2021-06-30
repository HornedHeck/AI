import torch
from torch import nn


class AEModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.e_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.e_linear = nn.Sequential(
            nn.Linear(in_features=784, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=20),
            nn.ReLU(),
        )
        self.d_linear = nn.Sequential(
            nn.Linear(in_features=20, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=784),
            nn.ReLU(),
        )
        self.d_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(2, 2), stride=(2, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.e_conv(x)
        x = torch.flatten(x, 1)
        x = self.e_linear(x)
        x = self.d_linear(x)
        x = x.view(-1, 16, 7, 7)
        return self.d_conv(x)

    def decode(self, x):
        x = self.d_linear(x)
        x = x.view(-1, 16, 7, 7)
        return self.d_conv(x)

    def encode(self, x):
        x = self.e_conv(x)
        x = torch.flatten(x, 1)
        return self.e_linear(x)


def get_model():
    return AEModel()


def get_simple_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5, 5), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(2, 2), stride=(2, 2)),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(2, 2), stride=(2, 2)),
        nn.Sigmoid()
    )
