import time

import numpy as nmp
import pandas
import torch

from music_generator_v1.MRNN_V4 import MRNN
from musicnet import MusicNet

path = '/home/hornedheck/PycharmProjects/AI/models/mrnn_v4_0.569_f.pt'
res_path = f'{int(time.time())}.csv'
note_len = MusicNet.note_struct_len


def array_to_note_struct(arr: nmp.array):
    arr = nmp.reshape(arr, note_len)
    start_delta = arr[0]
    duration = arr[1]
    note = arr[2:].argmax() + 10
    return [1, start_delta, duration, note]


model = MRNN(note_len, MusicNet.note_len, 1)
model.load_state_dict(torch.load(path))
start = torch.randn(1000 * note_len)
df = pandas.DataFrame(columns=['instrument', 'start', 'duration', 'note'])
df.reset_index(drop=True, inplace=True)
with torch.no_grad():
    res = model(start)
    j = 0
    for r in res:
        df.loc[j] = array_to_note_struct(r.numpy())
        j += 1

df.to_csv(res_path)
