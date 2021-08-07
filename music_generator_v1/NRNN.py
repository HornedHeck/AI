import torch
from torch import nn, Tensor

from base.BaseModel import BaseModel


class NRNN(BaseModel):

    def __init__(self, note_classes, minibatch_size: int):
        super().__init__()
        self.__loss = nn.CrossEntropyLoss()
        self.lstm_size = note_classes * 2
        self.minibatch_size = minibatch_size
        self.note_classes = note_classes
        self.embedding = nn.Embedding(note_classes, self.lstm_size)
        self.rec = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size)
        self.hc1 = None

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=note_classes * 2, out_features=note_classes * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2 * note_classes, out_features=note_classes),
        )

        self.clear()
        self.__batch__ = 0

    def forward(self, x: Tensor, hc: tuple[Tensor]) -> Tensor:
        x = self.embedding(x)
        x = x.view(1, self.minibatch_size, self.lstm_size)
        x, hc = self.rec(x.float(), hc)
        x = self.linear(x)
        return x.view(-1, self.note_classes), hc

    def loss(self, r: Tensor, y: Tensor) -> Tensor:
        return self.__loss(r, y)

    @property
    def name(self) -> str:
        return 'nrrn'

    def is_correct(self, r: Tensor, y: Tensor) -> int:
        predicted = torch.max(r.data, 1)[1]
        return (predicted == y).sum()

    def zero_state(self, device):
        return (
            torch.randn(1, self.minibatch_size, self.lstm_size).to(device),
            torch.randn(1, self.minibatch_size, self.lstm_size).to(device)
        )

    def clear_grad(self):
        self.hc1 = repackage_hidden(self.hc1)


def repackage_hidden(
        h
        # , device
):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Tensor:
        t = h.data.clone().detach()
        # if not t.is_cuda:
        #     t = t.to(device)
        return t
    else:
        return tuple(repackage_hidden(
            v
            # , device
        ) for v in h)
