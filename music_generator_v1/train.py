import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from base.RNNTrainer import RNNTrainer
from music_generator_v1.MRNN import MRNN
from music_generator_v1.musicnet import MusicNet

mini_batch_size = 10
seq_len = 100
batch_size = 1
log_interval = 50
fast_load = False
# fast_load = True

test_set = MusicNet(
    '../dataset/music/beethoven/',
    seq_len,
    train=False,
    minibatch_len=mini_batch_size,
    fast_load=fast_load
)
test_loader = DataLoader(test_set, batch_size=batch_size)
note_struct_size = test_set.note_struct_len
note_size = test_set.note_len
duration_size = test_set.size_len
start_size = test_set.start_len

train_set = MusicNet(
    '../dataset/music/beethoven/',
    seq_len,
    train=True,
    minibatch_len=mini_batch_size,
    fast_load=fast_load
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# train_set = test_set
# train_loader = test_loader
# model = NRNN(test_set.note_len, mini_batch_size)
model = MRNN(
    test_set.note_len,
    test_set.size_len,
    test_set.start_len,
    mini_batch_size
)
device = torch.device('cuda')
trainer = RNNTrainer(
    model,
    train_loader,
    test_loader,
    Adam(model.parameters(), 0.00001),
    device,
    safe_model=True,
    log_interval=log_interval,
    batches=1000,
    test_batches=200
)

trainer.train(50)
