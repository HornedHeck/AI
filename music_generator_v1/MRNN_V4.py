import torch
from torch import nn, Tensor


def repackage_hidden(h, device):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Tensor:
        t = h.data.clone().detach()
        if not t.is_cuda:
            t = t.to(device)
        return t
    else:
        return tuple(repackage_hidden(v, device) for v in h)


class MRNN(nn.Module):

    def __init__(self, note_struct_size: int, note_size: int, minibatch_size: int):
        super().__init__()
        self.note_struct_size = note_struct_size
        self.note_size = note_size
        self.t_size = note_struct_size - note_size

        self.rnn_n = MRNN_Rec(note_struct_size, minibatch_size)
        self.rnn_t = MRNN_Rec(note_struct_size, minibatch_size)
        self.l1 = self.rnn_n.l1
        self.l2 = self.rnn_n.l2
        self.l3 = self.rnn_n.l3

        self.l_n = nn.Sequential(
            nn.Linear(in_features=self.l1 + self.l2 + self.l3, out_features=4 * note_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4 * note_size, out_features=2 * note_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2 * note_size, out_features=note_size),
            nn.LogSoftmax()
        )
        self.l_t = nn.Sequential(
            nn.Linear(in_features=self.l1 + self.l2 + self.l3, out_features=4 * self.t_size),
            nn.ReLU(),
            nn.Linear(in_features=4 * self.t_size, out_features=2 * self.t_size),
            nn.ReLU(),
            nn.Linear(in_features=2 * self.t_size, out_features=self.t_size),
            nn.ReLU(),
        )

    def reset_hidden(self):
        self.rnn_n.reset_hidden()
        self.rnn_t.reset_hidden()

    def repackage_hidden(self, device):
        self.rnn_n.repackage_hidden(device)
        self.rnn_t.repackage_hidden(device)

    def forward(self, x):
        n_x = self.rnn_n(x)
        n_x = self.l_n(n_x)
        t_x = self.rnn_t(x)
        t_x = self.l_t(t_x)
        return torch.cat((t_x, n_x), -1)


class MRNN_Rec(nn.Module):

    def __init__(self, size: int, minibatch_size: int):
        super().__init__()
        self.minibatch_size = minibatch_size
        self.note_struct_size = size

        self.hc1 = None
        self.hc2 = None
        self.hc3 = None

        self.l1 = size + 2
        self.lstm_1 = nn.LSTM(input_size=size, hidden_size=self.l1)

        self.l2 = self.l1 + 4
        self.lstm_2 = nn.LSTM(input_size=self.l1, hidden_size=self.l2)

        self.l3 = self.l2 + 8
        self.lstm_3 = nn.LSTM(input_size=self.l2, hidden_size=self.l3)

        self.reset_hidden()

    def forward(self, x):
        x = x.view(-1, self.minibatch_size, self.note_struct_size)

        l1, self.hc1 = self.lstm_1(x, self.hc1)
        l2, self.hc2 = self.lstm_2(l1, self.hc2)
        l3, self.hc3 = self.lstm_3(l2, self.hc3)
        x = torch.cat((l1, l2, l3), dim=2)
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

    def repackage_hidden(self, device):
        self.hc1 = repackage_hidden(self.hc1, device)
        self.hc2 = repackage_hidden(self.hc2, device)
        self.hc3 = repackage_hidden(self.hc3, device)
