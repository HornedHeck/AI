import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import model


def visualize_mnist_model(model: nn.Module, batch_size: int = 10):
    model.to(torch.device("cpu"))
    test_loader = DataLoader(
        MNIST('../dataset', train=False, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(
                      (0.1307,), (0.3081,))
              ])),
        batch_size=batch_size, shuffle=True)

    correct = 0
    model.eval()
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x = Variable(x_b).float()
            y = Variable(y_b)
            predicted = torch.max(model(x), 1)[1]

            for i in range(batch_size):
                if y[i] == predicted[i]:
                    correct += 1
                else:
                    plt.title(f'Expected: {y[i]}\nGot: {predicted[i]}')
                    plt.imshow(x[i].view(28, 28))
                    plt.show()

    # noinspection PyTypeChecker
    print(f'Accuracy: {100. * correct / len(test_loader.dataset)}%')


if __name__ == '__main__':
    model = model.get_model()
    model.load_state_dict(torch.load('../models/conv_mnist_5_0.9917.pt'))
    visualize_mnist_model(model)
