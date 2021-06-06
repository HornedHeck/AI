import numpy as nmp
import pygad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Simple NL to calculate a xor b
# Input Layer 2: a, b
# Output Layer 1: value
# Expected Function: a xor b

DATASET_SIZE = 100
EPOCHS = 50
LOG_INTERVAL = DATASET_SIZE // 10


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 3 layers: 3n, 4n, 1n
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x


def generate_dataset() -> (nmp.ndarray, nmp.ndarray):
    data = nmp.random.randint(0, 2, (DATASET_SIZE, 2))
    res = data[:, 0] ^ data[:, 1]
    return data, res


dataset, expected = generate_dataset()
dataset = torch.from_numpy(dataset)
expected = torch.from_numpy(expected)
print(dataset)
print(expected)

net = Net().float()
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.2)
criterion = nn.MSELoss()

for e in range(EPOCHS):
    for i in range(DATASET_SIZE):
        data = Variable(dataset[i])
        target = Variable(expected[i])
        optimizer.zero_grad()
        net_res = net(data.float())
        loss = criterion(net_res, target.float())
        loss.backward()
        optimizer.step()
        print(f"Epoch {e} [{(i + 1)}/{DATASET_SIZE} ({i + 1})%] Loss: {loss.item()}")

test = torch.Tensor(4, 2)
test[1, 1] = 1
test[2, 0] = 1
test[3, 0] = 1
test[3, 1] = 1
print(torch.round(test))
print(torch.round(net(test)))


