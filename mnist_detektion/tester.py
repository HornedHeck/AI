import matplotlib.pyplot as plt
import torch.utils.data
from torch import nn, optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import bare_model
import relu_model
import conv_model
import norm_relu_model
import extended_norm_relu_model
from general import BATCH_SIZE, TRAIN_EPOCH_COUNT, EPOCH_BATCH_COUNT, MNIST_DATA_SIZE, interpolate_plot, EPOCH_SIZE

import numpy as nmp

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_data = MNIST(
    "../dataset",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=BATCH_SIZE, shuffle=True)
train_loader_iter = iter(train_loader)

models = [
    conv_model.get_model(),
    extended_norm_relu_model.get_model(),
    norm_relu_model.get_model(),
    relu_model.get_model(),
    bare_model.get_model()
]

learning_rate = 0.001
loss = nn.CrossEntropyLoss()
optimizers = [
    optim.Adam(models[0].parameters(), lr=learning_rate),
    optim.Adam(models[1].parameters(), lr=learning_rate),
    optim.Adam(models[2].parameters(), lr=learning_rate),
    optim.Adam(models[3].parameters(), lr=learning_rate),
    optim.Adam(models[4].parameters(), lr=learning_rate),
]

epoch_count = TRAIN_EPOCH_COUNT

epochs_losses = []

for epoch in range(epoch_count):
    epoch_losses = []
    predicted = []

    for model in models:
        epoch_losses.append([])
        predicted.append(0)
        model.train()

    for i in range(EPOCH_BATCH_COUNT):
        x, y = next(train_loader_iter)
        for k in range(len(models)):
            optimizers[k].zero_grad()
            res = models[k](x.view(-1, MNIST_DATA_SIZE))
            loss_value = loss(res, y)
            loss_value.backward()
            epoch_losses[k].append(loss_value.item())
            optimizers[k].step()
            res_p = torch.argmax(res, dim=1)
            predicted[k] += nmp.count_nonzero(res_p == y)

    predicted = nmp.array(predicted, dtype=float)

    epoch_losses = nmp.average(epoch_losses, axis=1)
    print(f"Epoch {epoch}/{TRAIN_EPOCH_COUNT}. Losses: {epoch_losses} Acc: {predicted * 100 / EPOCH_SIZE}")
    epochs_losses.append(epoch_losses)

epochs_losses = nmp.transpose(epochs_losses)

names = ['Conv2D', 'Extended', 'Norm ReLu', 'ReLu', 'Bare']

for i in range(len(epochs_losses)):
    interpolate_plot(epochs_losses[i], epoch_count, names[i])
plt.legend(loc='best')
plt.show()
#
# test_x, test_y = next(train_loader_iter)
# res = model(test_x.view(-1, MNIST_DATA_SIZE)).detach()
# res = torch.softmax(res, 0)
# for i in range(len(res)):
#     print(f"{i + 1}. Expected: {test_y[i]} got: {nmp.argmax(res[i])}")
#     plt.title(f"Sample №{i + 1}")
#     plt.imshow(test_x[i].view(28, 28))
#     plt.show()
