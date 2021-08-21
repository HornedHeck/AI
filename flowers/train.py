import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from base.BaseTrainer import BaseTrainer
from flowers.cnn import FlowersCNN
from flowers.dataset import UKFlowers, Type

batch_size = 10
log_interval = 50
root = '''/home/hornedheck/PycharmProjects/AI/dataset/flowers'''

print('Start loading data.')
val_set = UKFlowers(root, Type.VAL, False)
val_loader = DataLoader(val_set, batch_size=batch_size)
print('Validation data ready.')

train_set = UKFlowers(root, Type.TRAIN, False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# train_set = val_set
# train_loader = val_loader
print('Train data ready.')

model = FlowersCNN()
device = torch.device('cuda')
trainer = BaseTrainer(
    model,
    train_loader,
    val_loader,
    Adam(model.parameters(), 0.001),
    device,
    safe_model=True,
    log_interval=log_interval,
    # batches=1000,
    # test_batches=200
)

trainer.train(50)
