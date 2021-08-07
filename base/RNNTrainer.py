import torch
from torch import Tensor

from base.BaseTrainer import BaseTrainer


class RNNTrainer(BaseTrainer):

    def train_batch(self, x: Tensor, y: Tensor):
        self.opt.zero_grad()

        x = x.view(-1, *x.shape[2:])
        y = y.view(-1, y.shape[2])
        hc = self.model.zero_state(self.device)
        for i in range(0, x.shape[1]):
            res, hc = self.model(x[:, i], hc)

        error = self.model.loss(res, y)

        hc[0].detach()
        hc[1].detach()

        error.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.opt.step()

        return error

    def test_batch(self, x: Tensor):
        x = x.view(-1, *x.shape[2:])
        hc = self.model.zero_state(self.device)
        for i in range(0, x.shape[1]):
            res, hc = self.model(x[:, i], hc)

        return res
