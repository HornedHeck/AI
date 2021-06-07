import copy
import random
import time

import torch
from torch import nn, FloatTensor, sigmoid, Tensor

from flappy_bird.actor import Actor

POPULATION_SIZE = 32


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.fitness = 0
        self.fc1 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1.forward(x)
        return sigmoid(x)


class NeuroActor(Actor):

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.fitness = int(time.time() * 100)
        self.model = model

    def is_up(self, t_h: int, b_h: int, v_speed: int, to_pipe: int):
        tensor = FloatTensor([t_h, b_h, v_speed, to_pipe])
        return self.model.forward(tensor)[0] >= 0.5

    def finish(self):
        self.fitness = int(time.time() * 100) - self.fitness
        self.model.fitness = self.fitness
        finished__generation.append(self.model)
        # print(f"Neuro Actor finished with fitness {self.fitness}")


finished__generation = []


def get_chromosomes(model: Model) -> (Tensor, Tensor):
    weights = model.fc1.weight
    m = weights.shape[1] // 2
    return weights[0, :m], weights[0, m:]


def mutate(model: Model):
    index = random.randint(-1, model.fc1.weight.shape[1] - 1)
    if index < 0:
        model.fc1.bias[0] += random.uniform(-.8, .8) * model.fc1.bias[0]
    else:
        model.fc1.weight[0, index] += random.uniform(-1.25, 1.25) * model.fc1.weight[0, index]


def generate_pack(bias: float, start_weight: Tensor):
    base = Model()
    base.fc1.weight = nn.Parameter(
        torch.reshape(start_weight, base.fc1.weight.shape)
    )
    base.fc1.bias[0] = bias
    res = [base]
    for i in range(POPULATION_SIZE // 8 - 1):
        mutant = copy.deepcopy(base)
        mutate(mutant)
        res.append(mutant)

    return res


def new_generation():
    p1, p2 = finished__generation[-2:]
    finished__generation.clear()
    bias = p1.fc1.bias[0].item()

    p1_l, p1_h = get_chromosomes(p1)
    p2_l, p2_h = get_chromosomes(p2)

    g111 = generate_pack(bias, torch.hstack((p1_l, p1_h)))
    g112 = generate_pack(bias, torch.hstack((p1_l, p2_h)))
    g121 = generate_pack(bias, torch.hstack((p2_l, p1_h)))
    g122 = generate_pack(bias, torch.hstack((p2_l, p2_h)))

    bias = p2.fc1.bias[0].item()
    g211 = generate_pack(bias, torch.hstack((p1_l, p1_h)))
    g212 = generate_pack(bias, torch.hstack((p1_l, p2_h)))
    g221 = generate_pack(bias, torch.hstack((p2_l, p1_h)))
    g222 = generate_pack(bias, torch.hstack((p2_l, p2_h)))

    return list(set().union(g111, g112, g121, g122, g211, g212, g221, g222))


def base_generation():
    res = []
    for i in range(POPULATION_SIZE):
        res.append(Model())
    return res


if __name__ == '__main__':
    src = Model()
    src_w1, src_w2 = get_chromosomes(src)
    pack = generate_pack(torch.hstack((src_w1, src_w2)))
    for m in pack:
        print(m.fc1.weight)
    pass
