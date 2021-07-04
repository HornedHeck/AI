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
    raw_folder = 'raw'
    train_labels = 'train_labels'
    test_labels = 'test_labels'
    note_len = 95
    note_struct_len = note_len + 2
    note_offset = 10

    @staticmethod
    def note_to_one_hot(note: int):
        zeros = nmp.zeros(MusicNet.note_len, dtype=float)
        zeros[note - MusicNet.note_offset] = 1.
        return zeros

    def __init__(self, root, train=True, batch_size: int = 250, transforms=None):
        self.root = root
        self.transforms = transforms
        self.batch_size = batch_size
        self.note_length = 95
        self.note_offset = 10
        self.notes = dict()
        if train:
            labels_path = os.path.join(self.root, self.train_labels)
        else:
            labels_path = os.path.join(self.root, self.test_labels)

        self.labels = self.process_labels(labels_path)

    def __getitem__(self, index) -> T_co:
        item = self.labels[:, index]
        return item[0], item[1]

    def __len__(self):
        return self.labels.shape[1] // self.batch_size * self.batch_size

    def process_labels(self, path):
        trees = dict()
        for item in os.listdir(path):
            if not item.endswith('.csv'):
                continue
            uid = int(item[:-4])
            track = []
            with open(os.path.join(path, item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    if note_value not in self.notes.keys():
                        self.notes[note_value] = set()
                    self.notes[note_value].add(end_beat)
                    track.append((
                        # start_time, end_time, instrument,
                        start_beat, end_beat, *self.note_to_one_hot(note)
                    ))
            track = nmp.array(track)
            track[:-1, 0] = track[1:, 0] - track[:-1, 0]
            track = track[nmp.logical_and(track[:, 0] <= 4., track[:, 1] <= 4.)]
            track[track[:, 0] < 0., 0] = 0.
            trees[uid] = track[:len(track) // self.batch_size * self.batch_size + 1]
        return nmp.concatenate(tuple((notes[:-1], notes[1:]) for notes in trees.values()), axis=1)


# max = 104
# min = 21
# used_min = 10
if __name__ == '__main__':
    dataset = MusicNet('../dataset/music/', train=True)
    print(nmp.unique(dataset.labels[:, :, 0]))
    # for k in dataset.notes.keys():
    #     print(f'{k}: {dataset.notes[k]}')
