import os.path

import torch.nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import model

batch_size = 10
log_interval = 50

train_loader = DataLoader(
    MNIST('../dataset', train=True, download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(
                  (0.1307,), (0.3081,))
          ])),
    batch_size=batch_size, shuffle=True)

test_loader = DataLoader(
    MNIST('../dataset', train=False, download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(
                  (0.1307,), (0.3081,))
          ])),
    batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0")

epochs = 10
model = model.get_model().to(device)

base_path = '../models/'
os.makedirs(os.path.dirname(base_path), exist_ok=True)

optimizer = Adam(model.parameters(), 0.00001)
# optimizer = torch.optim.SGD(model.parameters(), 0.01)
loss = torch.nn.NLLLoss()

for e in range(epochs):
    model.train()
    # noinspection PyRedeclaration
    correct = 0
    for i, (x, y) in enumerate(train_loader):
        x_b = Variable(x).float().to(device)
        y_b = Variable(y).to(device)
        optimizer.zero_grad()
        res = model(x_b)
        error = loss(res, y_b)
        error.backward()
        optimizer.step()

        predicted = torch.max(res.data, 1)[1]
        correct += (predicted == y_b).sum()
        if i % log_interval == 0:
            # noinspection PyTypeChecker
            print(
                f'Epoch: {e + 1} [{i * batch_size}/{len(train_loader.dataset)}({100. * i / len(train_loader):.1f}%)]\t'
                f'Loss: {error.item():.6f}\t'
                f'Accuracy:{100. * correct / batch_size / log_interval:.2f}%')
            correct = 0

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x_b = Variable(x).float().to(device)
            y_b = Variable(y).to(device)
            res = model(x_b)
            predicted = torch.max(res.data, 1)[1]
            correct += (predicted == y_b).sum()

    # noinspection PyTypeChecker
    acc = correct / len(test_loader.dataset)
    print(
        f'Epoch: {e + 1} [Assert]\t'
        f'Accuracy: {100. * acc:.2f}%'
    )
    torch.save(model.state_dict(), f'{base_path}conv_mnist_{e + 1}_{acc:.4f}.pt')
