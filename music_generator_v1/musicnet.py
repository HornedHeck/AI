from __future__ import print_function

import csv
import os
import os.path

import numpy as nmp
import torch.utils.data as data
from torch.utils.data.dataset import T_co


# todo delta_start to classes
# todo duration to classes:
class MusicNet(data.Dataset):
    url = 'https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz'
    nmp_folder = 'numpy'
    train_labels = 'train_labels'
    test_labels = 'test_labels'

    note_len = 95
    note_offset = 10
    size_classes = nmp.array([
        0.125, 0.25, 0.375, 0.5, 0.75, 1., 1.125, 1.25, 1.5, 2., 3., 4.
    ])

    start_classes = nmp.array([
        0, 0.125, 0.25, 0.375, 0.5, 0.75, 1., 1.125, 1.25, 1.5, 2., 3., 4.
    ])

    instrument_classes = nmp.array([
        1., 7., 41., 42., 43., 44., 61., 69., 71., 72., 74.
    ])

    def get_size_class(self, size):
        return nmp.abs(self.size_classes - size).argmin()

    def get_start_class(self, start):
        return nmp.abs(self.start_classes - start).argmin()

    def get_instr_class(self, instr):
        return nmp.abs(self.instrument_classes - instr).argmin()

    size_len = len(size_classes)
    start_len = len(start_classes)
    instrument_len = len(instrument_classes)
    note_struct_len = note_len + size_len + start_len

    def __init__(self, root, seq_len: int, train=True, fast_load: bool = True, minibatch_len: int = 25):
        self.root = root
        self.minibatch_len = minibatch_len
        self.seq_len = seq_len
        if not fast_load:
            if train:
                labels_path = os.path.join(self.root, self.train_labels)
                save_path = os.path.join(self.root, self.nmp_folder, self.train_labels)
            else:
                labels_path = os.path.join(self.root, self.test_labels)
                save_path = os.path.join(self.root, self.nmp_folder, self.test_labels)

            labels = self.process_labels(labels_path)
            save_path += '.npy'

            nmp.save(save_path, labels)

        if train:
            labels_path = os.path.join(self.root, self.nmp_folder, self.train_labels)
        else:
            labels_path = os.path.join(self.root, self.nmp_folder, self.test_labels)
        labels_path += '.npy'
        self.x, self.y = self.process_fast_labels(labels_path)

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def process_fast_labels(self, path):
        raw = nmp.load(path, allow_pickle=True)
        x = []
        y = []
        for row in raw:
            row[:, 1] = nmp.array([self.get_size_class(s) for s in row[:, 1]])
            row[:, 2] = nmp.array([self.get_start_class(s) for s in row[:, 2]])
            # row[:, 3] = nmp.array([self.get_instr_class(s) for s in row[:, 3]])
            row = row.astype(int)
            s_idx = (row.shape[0] - 1) // self.seq_len * self.seq_len + 1
            for i in range(s_idx - self.seq_len - 1):
                x.append(row[i: i + self.seq_len])
                y.append(row[i + self.seq_len + 1])
        x = nmp.array(x)
        y = nmp.array(y)
        batch = self.minibatch_len * self.seq_len
        batch_count = y.shape[0] // batch * batch
        x = x[:batch_count]
        x = x.reshape(-1, self.minibatch_len, self.seq_len, x.shape[-1])
        y = y[:batch_count]
        y = y.reshape((-1, self.minibatch_len, y.shape[-1]))
        return x, y

    def process_labels(self, path):
        trees = list()
        for item in os.listdir(path):
            if not item.endswith('.csv'):
                continue
            track = []
            with open(os.path.join(path, item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    # start_time = int(label['start_time'])
                    # end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note']) - self.note_offset
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    # note_value = label['note_value']
                    track.append(
                        [
                            note,
                            end_beat,
                            start_beat,
                            # instrument
                        ]
                    )
            track = nmp.array(track)
            track[:-1, 2] = track[1:, 2] - track[:-1, 2]
            trees.append(track[:-1])
        return nmp.array(trees)


# max = 104
# min = 21
# used_min = 10
if __name__ == '__main__':
    dataset = MusicNet('/home/hornedheck/PycharmProjects/AI/dataset/music/beethoven/', 100, train=True, fast_load=True)
    print(dataset)
