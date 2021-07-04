import torch
from torch import nn


class MRNN(nn.Module):

    def __init__(self, note_struct_size: int, minibatch_size: int):
        super().__init__()
        self.minibatch_size = minibatch_size
        self.note_struct_size = note_struct_size

        self.hc1 = None
        self.l1 = note_struct_size * 2
        self.lstm_1 = nn.LSTM(input_size=note_struct_size, hidden_size=self.l1, num_layers=2)
        self.reset_hidden()

        self.l = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=self.l1, out_features=2 * note_struct_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2 * note_struct_size, out_features=note_struct_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.minibatch_size, self.note_struct_size)

        x, self.hc1 = self.lstm_1(x, self.hc1)
        x = self.l(x)
        return x

    def reset_hidden(self):
        self.hc1 = (
            torch.ones(2, self.minibatch_size, self.l1),
            torch.ones(2, self.minibatch_size, self.l1)
        )
