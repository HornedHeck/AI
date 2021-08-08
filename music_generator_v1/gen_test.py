import time

import numpy as nmp
import torch
from torch import tensor, Tensor

from music_generator_v1.MRNN import MRNN
from musicnet import MusicNet

path = '/home/hornedheck/PycharmProjects/AI/models/mrrn_0.120_15_34_35.pt'
res_path = f'{int(time.time())}.csv'
note_len = MusicNet.note_len


def array_to_note_struct(arr: nmp.array):
    arr = nmp.reshape(arr, note_len)
    start_delta = MusicNet.size_classes[arr[:MusicNet.size_len].argmax()]
    duration = MusicNet.size_classes[arr[MusicNet.size_len:2 * MusicNet.size_len].argmax()]
    note = arr[2 * MusicNet.size_len:].argmax() + 10
    return [1, start_delta, duration, note]


# model = MRNN(note_len, 1)
model = MRNN(
    MusicNet.note_len,
    MusicNet.size_len,
    MusicNet.start_len,
    MusicNet.instrument_len,
    1
)
model.load_state_dict(torch.load(path))
device = torch.device('cpu')
notes = nmp.arange(note_len)
sizes = nmp.arange(MusicNet.size_len)
starts = nmp.arange(MusicNet.start_len)
instruments = nmp.arange(MusicNet.instrument_len)
note = tensor([60, 0, 0, 0], device=device).view(1, 4)
hc = model.zero_state(device)


def generate(items: nmp.ndarray, p: Tensor):
    p = p / p.sum()
    return torch.tensor(nmp.random.choice(items, p=p.numpy()), device=device)


with torch.no_grad():
    for i in range(300):
        note, hc = model(note, hc)
        note = torch.softmax(note, 1)[0]

        start = 0
        end = MusicNet.note_len
        n = generate(notes, note[start:end])
        start = end
        end += MusicNet.size_len
        size = generate(sizes, note[start:end])
        start = end
        end += MusicNet.start_len
        offset = generate(starts, note[start:end])
        start = end
        end += MusicNet.instrument_len
        instrument = generate(instruments, note[start:end])

        note = torch.stack((n, size, offset, instrument)).view(1, 4)

        print(
            f'{n.item() + 10} '
            f'{MusicNet.size_classes[size.item()]} '
            f'{MusicNet.start_classes[offset.item()]} '
            f'{MusicNet.instrument_classes[instrument.item()]},',
            end=''
        )

# df = pandas.DataFrame(columns=['instrument', 'start', 'duration', 'note'])
# df.reset_index(drop=True, inplace=True)
# with torch.no_grad():
#     for i in range(1000):
#         start = model(start)
#         df.loc[i] = array_to_note_struct(start.numpy())
#
# df.to_csv(res_path)
