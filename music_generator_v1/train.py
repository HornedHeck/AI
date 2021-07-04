import errno
import os
import signal
from time import time

import numpy as nmp
from torch.utils.data import DataLoader

from music_generator_v1.musicnet import MusicNet

checkpoint_path = './checkpoints'
checkpoint = 'musicnet_demo.pt'

try:
    os.makedirs(checkpoint_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def worker_init(args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore signals so parent can handle them
    nmp.random.seed(os.getpid() ^ int(time()))  # approximately random seed for workers


batch_size = 100
kwargs = {'num_workers': 4, 'pin_memory': True, 'worker_init_fn': worker_init}

# train_set = MusicNet('../dataset/music/', train=True, download=True, mmap=False)
# train_loader = DataLoader(train_set, batch_size=100 , **kwargs)

test_set = MusicNet('../dataset/music/', train=False, download=True)
# test_loader = DataLoader(test_set, batch_size=10, **kwargs)
print(len(test_set.labels))
# with test_set:
#     it = iter(test_loader)
#     x, y = next(it)
#     for y_i in y:
#         print((y_i == 1).nonzero(as_tuple=True))
#     for i, (x, y) in enumerate(test_loader):
#         zeros = torch.count_nonzero(y, dim=1)
#         print(zeros)
#         if i == 10:
#             break
