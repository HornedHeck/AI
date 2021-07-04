import torch
from torch import nn


class MRNN(nn.Module):

    def __init__(self, note_struct_size: int, minibatch_size: int):
        super().__init__()
        self.minibatch_size = minibatch_size
        self.note_struct_size = note_struct_size
        self.hc1 = None
        self.hc2 = None
        self.hc3 = None
        self.l1 = note_struct_size + 1
        self.lstm_1 = nn.LSTM(input_size=note_struct_size, hidden_size=self.l1)

        self.l2 = self.l1 + 1
        self.lstm_2 = nn.LSTM(input_size=self.l1, hidden_size=self.l2)

        self.l3 = self.l2 + 1
        self.lstm_3 = nn.LSTM(input_size=self.l2, hidden_size=self.l3)

        self.reset_hidden()
        self.l = nn.Sequential(
            nn.Linear(in_features=self.l1 + self.l2 + self.l3, out_features=4 * note_struct_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4 * note_struct_size, out_features=2 * note_struct_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2 * note_struct_size, out_features=note_struct_size)
        )

    def forward(self, x):
        x = x.view(-1, self.minibatch_size, self.note_struct_size)

        l1, self.hc1 = self.lstm_1(x, self.hc1)
        l2, self.hc2 = self.lstm_2(l1, self.hc2)
        l3, self.hc3 = self.lstm_3(l2, self.hc3)
        x = self.l(torch.cat((l1, l2, l3), dim=2))
        return x

    def reset_hidden(self):
        self.hc1 = (
            torch.randn(1, self.minibatch_size, self.l1),
            torch.randn(1, self.minibatch_size, self.l1)
        )
        self.hc2 = (
            torch.randn(1, self.minibatch_size, self.l2),
            torch.randn(1, self.minibatch_size, self.l2)
        )
        self.hc3 = (
            torch.randn(1, self.minibatch_size, self.l3),
            torch.randn(1, self.minibatch_size, self.l3)
        )
