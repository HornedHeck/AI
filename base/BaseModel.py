import abc

from torch import nn, Tensor


class BaseModel(nn.Module, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def loss(self, r: Tensor, y: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def is_correct(self, r: Tensor, y: Tensor) -> int:
        pass

    def clear(self):
        pass

    def clear_grad(self):
        pass
