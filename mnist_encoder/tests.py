import random

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as nmp

from model import get_model, get_simple_model, AEModel


def noise(img):
    coords = nmp.random.randint(0, 27, (70, 2))
    for x, y in coords:
        img[0, x, y] = 0.5
    return img


path_simple = '/home/hornedheck/PycharmProjects/AI/models/autoencoder_mnist_simple_1_0.0088.pt'
path = '/home/hornedheck/PycharmProjects/AI/models/autoencoder_mnist_25_0.0111.pt'


def test_mode_1(model: torch.nn.Module):
    test_loader = DataLoader(
        MNIST('../dataset', train=False, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
              ])),
        batch_size=10, shuffle=True)

    data, _ = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        for image in data:
            f, ax = plt.subplots(1, 2)
            image = noise(image)
            ax[0].imshow(image.view(28, 28))
            ax[1].imshow(model(image.view(1, 1, 28, 28)).view(28, 28))
            plt.show()


deviations_name = 'std.npy'
avg_name = 'avg.npy'


def init_base():
    base_deviation = nmp.load(deviations_name)
    base_values = nmp.load(avg_name)
    return base_values, base_deviation


def generate_mode_2_data(digit: int, base_values, base_deviation):
    values = base_values[digit]
    deviations = base_deviation[digit]
    k = nmp.random.random(values.shape) - 0.5
    return torch.tensor(values + deviations * k, dtype=torch.float32)
    pass


def test_mode_2(model: AEModel):
    model.eval()
    avg, d = init_base()
    with torch.no_grad():
        for i in range(10):
            plt.title(f'{i}')
            plt.imshow(model.decode(
                generate_mode_2_data(2, avg, d)
            ).view(28, 28))
            plt.show()


def create_params():
    model = get_model()
    model.load_state_dict(torch.load(path))
    values = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]

    test_loader = DataLoader(
        MNIST('../dataset', train=False, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
              ])),
        batch_size=1, shuffle=True)

    for x, y in test_loader:
        values[y.item()].append(model.encode(x).detach().numpy())

    nmp.save('values.npy', nmp.array([v[:500] for v in values]))


def research_params():
    values = nmp.load('values.npy').reshape((10, 500, 20))
    deviations = nmp.std(values, axis=1)
    nmp.save(deviations_name, deviations)
    average = nmp.average(values, axis=1)
    nmp.save(avg_name, average)


# research_params()
m = get_model()
m.load_state_dict(torch.load(path))
test_mode_2(m)
