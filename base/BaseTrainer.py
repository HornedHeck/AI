import time
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from base.BaseModel import BaseModel


class BaseTrainer(object):

    def __init__(
            self,
            model: BaseModel,
            train_loader: DataLoader,
            test_loader: DataLoader,
            optimizer: Optimizer,
            device: torch.device,
            safe_model: bool = True,
            log_interval: Optional[int] = None,
            batches: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.batches = batches
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.opt = optimizer
        self.model = model.to(device)
        self.safe = safe_model
        if log_interval is None:
            self.log_interval = len(train_loader) // 50
        else:
            self.log_interval = log_interval

    def train_batch(self, x: Tensor, y: Tensor):
        self.opt.zero_grad()
        res = self.model(x)
        error = self.model.loss(res, y)
        error.backward()
        self.opt.step()
        return error

    def test_batch(self, x: Tensor):
        return self.model(x)

    def __epoch__(self, e: int):
        self.model.train()
        batch_size = self.train_loader.batch_size
        batches = self.batches if self.batches is not None else len(self.train_loader)
        for i, (x, y) in enumerate(self.train_loader):
            error = self.train_batch(x.to(self.device), y.to(self.device))

            if (i + 1) % self.log_interval == 0:
                # noinspection PyTypeChecker
                print(
                    f'Epoch: {e + 1} [{(i + 1) * batch_size}/'
                    f'{batches * batch_size}'
                    f'({100. * (i + 1) / batches:.1f}%)]\t'
                    f'Loss: {error.item():.4f}'
                )

            if self.batches == i:
                break

        self.model.eval()
        print(f'Epoch: {e + 1} [Assert started].')
        correct = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                r = self.test_batch(x.to(self.device))
                correct += self.model.is_correct(r, y.to(self.device))
                if i == 19:
                    break

        # noinspection PyTypeChecker
        acc = correct / 20 * batch_size
        print(
            f'Epoch: {e + 1} [Assert]\t'
            f'Accuracy: {100. * acc:.2f}%'
        )

        if self.safe:
            name = \
                f'{self.model.name}_' \
                f'{acc:.3f}_' \
                f'{time.strftime("%H_%M_%S", time.localtime(time.time()))}.pt'
            torch.save(
                self.model.state_dict(),
                f'../models/{name}'
            )

    def train(self, epochs: int):
        for e in range(epochs):
            self.__epoch__(e)
