import os.path

import torch
import torch.nn as nn
from torch import Tensor

from model import get_model
from torch.autograd import Variable
from torch.optim import Adagrad, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

batch_size = 10
log_interval = 50
epochs = 25
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')
torch.set_num_threads(14)

train_loader = DataLoader(
    MNIST('../dataset', train=True, download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
          ])),
    batch_size=batch_size, shuffle=True)

test_loader = DataLoader(
    MNIST('../dataset', train=False, download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
          ])),
    batch_size=batch_size, shuffle=True)

encoder = get_model()
encoder.load_state_dict(
    torch.load('/home/hornedheck/PycharmProjects/AI/models/autoencoder_mnist_25_0.0111.pt')
)

loss = nn.BCELoss()
opt = Adam(encoder.parameters(), 0.1)

for e in range(epochs):
    encoder.train()
    # noinspection PyRedeclaration
    for i, (x, _) in enumerate(train_loader):
        x_b = Variable(x).float().to(device)
        opt.zero_grad()
        res = encoder(x_b)
        error = loss(res, x_b)
        error.backward()
        opt.step()

        if i % log_interval == 0:
            # noinspection PyTypeChecker
            print(
                f'Epoch: {e + 1} [{i * batch_size}/{len(train_loader.dataset)}({100. * i / len(train_loader):.1f}%)]\t'
                f'Loss: {error.item():.6f}\t'
            )

    encoder.eval()
    error = 0.
    with torch.no_grad():
        for x, y in test_loader:
            x_b = Variable(x).float().to(device)
            res = encoder(x_b)
            error += loss(res, x_b).item()

    # noinspection PyTypeChecker
    acc = error / len(test_loader.dataset)
    print(
        f'Epoch: {e + 1} [Assert]\t'
        f'Average Loss: {error:.4f}'
    )
    torch.save(encoder.state_dict(),
               f'/home/hornedheck/PycharmProjects/AI/models/autoencoder_mnist_{e + 1}_{acc:.4f}.pt')
