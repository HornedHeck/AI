import torch
from torch import nn, Tensor
from torch.nn import Embedding, LSTM, Sequential

from base.BaseModel import BaseModel


# note, len, offset
class MRNN(BaseModel):

    def __init__(
            self,
            note_classes: int,
            len_classes: int,
            offset_classes: int,
            minibatch_size: int
    ):
        super().__init__()

        self.note_classes = note_classes
        self.len_classes = len_classes
        self.offset_classes = offset_classes

        self.__loss = nn.CrossEntropyLoss()
        self.minibatch_size = minibatch_size

        self.lstm_multiplier = 2
        self.base_size = (note_classes + len_classes + offset_classes)
        self.lstm_size = self.base_size * self.lstm_multiplier
        self.note_embedding = Embedding(note_classes, note_classes * self.lstm_multiplier)
        self.len_embedding = Embedding(len_classes, len_classes * self.lstm_multiplier)
        self.offset_embedding = Embedding(offset_classes, offset_classes * self.lstm_multiplier)

        self.lstm = LSTM(self.lstm_size, self.lstm_size)

        self.linear = Sequential(
            nn.Dropout(),
            nn.Linear(in_features=self.lstm_size, out_features=self.lstm_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=self.lstm_size, out_features=self.base_size)
        )

    def embed(self, x: Tensor) -> Tensor:
        note_e = self.note_embedding(x[:, 0])
        size_e = self.len_embedding(x[:, 1])
        offset_e = self.offset_embedding(x[:, 2])
        return torch.cat((note_e, size_e, offset_e), 1)

    def forward(self, x: Tensor, hc: tuple[Tensor]):
        x = self.embed(x)
        x = x.view(1, self.minibatch_size, self.lstm_size)
        x, hc = self.lstm(x, hc)
        x = self.linear(x)
        return x.view(-1, self.base_size), hc

    @property
    def name(self) -> str:
        return 'mrrn'

    def zero_state(self, device):
        return (
            torch.randn(1, self.minibatch_size, self.lstm_size).to(device),
            torch.randn(1, self.minibatch_size, self.lstm_size).to(device)
        )

    def loss(self, r: Tensor, y: Tensor) -> Tensor:
        start = 0
        end = self.note_classes
        note_error = self.__loss(
            r[:, start:end], y[:, 0]
        )
        start = end
        end += self.len_classes
        size_error = self.__loss(
            r[:, start:end], y[:, 1]
        )
        start = end
        end += self.offset_classes
        offset_error = self.__loss(
            r[:, start:end], y[:, 2]
        )

        return note_error  # + size_error + offset_error

    def is_correct(self, r: Tensor, y: Tensor) -> int:
        start = 0
        end = self.note_classes
        notes = r[:, start:end].argmax(1)
        start = end
        end += self.len_classes
        sizes = r[:, start:end].argmax(1)
        start = end
        end += self.offset_classes
        offsets = r[:, start:end].argmax(1)

        r = torch.stack((notes, sizes, offsets), 1)
        y = y.view(r.shape)

        correct = 0
        for y_row, r_row in zip(y, r):
            correct += (y_row[0] == r_row[0])  # .all().item()

        return correct

    def clear(self):
        super().clear()

    def clear_grad(self):
        super().clear_grad()
