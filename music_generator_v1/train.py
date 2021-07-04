import numpy as nmp
import torch
from torch import Tensor
from torch.nn import NLLLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from music_generator_v1.MRNN_V4 import MRNN
from music_generator_v1.musicnet import MusicNet

checkpoint_path = './checkpoints'
checkpoint = 'musicnet_demo.pt'

device = torch.device('cuda')


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Tensor:
        t = h.data.clone().detach()
        if not t.is_cuda:
            t = t.to(device)
        return t
    else:
        return tuple(repackage_hidden(v) for v in h)


transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 250
mini_batch_size = 25

log_interval = 40

test_set = MusicNet('../dataset/music/', train=False)
test_loader = DataLoader(test_set, batch_size=batch_size)
note_struct_size = test_set.note_struct_len

# train_set = MusicNet('../dataset/music/', train=True)
# train_loader = DataLoader(train_set, batch_size=batch_size)
train_set = test_set
train_loader = test_loader

l1_loss = MSELoss()
ce_loss = NLLLoss()
model = MRNN(note_struct_size, train_set.note_len, mini_batch_size).to(device)


def loss(r, y):
    tempo_error = l1_loss(r[:, :, :2], y[:, :, :2])
    note_error = ce_loss(r[:, :, 2:].view(-1, test_set.note_length),
                         y[:, :, 2:].view(-1, test_set.note_length).argmax(1))
    return tempo_error, note_error


optimizer = Adam(model.parameters(), 0.05)

e_count = 10
for e in range(e_count):
    model.train()
    for batch, (x, y) in enumerate(train_loader):
        y_p = nmp.reshape(y, (y.shape[0] // mini_batch_size, mini_batch_size, note_struct_size))
        x_p = nmp.reshape(x, (y.shape[0] // mini_batch_size, mini_batch_size, note_struct_size))
        model.reset_hidden()
        for i in range(x_p.shape[0]):
            x_b = torch.tensor(x_p[i], dtype=torch.float32).view(1, mini_batch_size, note_struct_size).to(device)
            y_b = torch.tensor(y_p[i], dtype=torch.float32).view(1, mini_batch_size, note_struct_size).to(device)
            optimizer.zero_grad()
            model.repackage_hidden(device)
            model.zero_grad()

            res = model(x_b)
            error = loss(res, y_b)
            error[0].backward(retain_graph=True)
            error[1].backward()
            optimizer.step()
        if (batch + 1) % log_interval == 0:
            print(
                f'Epoch {e + 1}/{e_count}: '
                f'{(batch + 1) * batch_size}/{len(train_set)}({100. * (batch + 1) / len(train_loader) :.1f}%)')

    model.reset_hidden()
    model.eval()
    errors = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_loader):
            y_p = nmp.reshape(y, (y.shape[0] // mini_batch_size, mini_batch_size, note_struct_size))
            x_p = nmp.reshape(x, (y.shape[0] // mini_batch_size, mini_batch_size, note_struct_size))
            model.repackage_hidden(device)
            for i in range(x_p.shape[0]):
                x_b = torch.tensor(x_p[i], dtype=torch.float32).view(1, mini_batch_size, note_struct_size).to(device)
                y_b = torch.tensor(y_p[i], dtype=torch.float32).view(1, mini_batch_size, note_struct_size).to(device)
                res = model(x_b)
                errors.append(loss(res, y_b)[0].item())

    errors = nmp.average(errors)
    torch.save(model.state_dict(), f'../models/mrnn_v4_{errors:.3f}.pt')
    print(f'Epoch {e + 1}/{e_count} acquire: {errors:.3f}')
