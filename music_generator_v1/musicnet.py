from __future__ import print_function

import csv
import errno
import os
import os.path
import pickle
from subprocess import call

import torch.utils.data as data
from intervaltree import IntervalTree
from scipy.io import wavfile

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization


class MusicNet(data.Dataset):
    """`MusicNet <http://homes.cs.washington.edu/~thickstn/musicnet.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``train_data``,
            otherwise from ``test_data``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        mmap (bool, optional): If true, mmap the dataset for faster access times.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        window (int, optional): Size in samples of a data point.
        pitch_shift (int,optional): Integral pitch-shifting transformations.
        jitter (int, optional): Continuous pitch-jitter transformations.
        epoch_size (int, optional): Designated Number of samples for an "epoch"
    """
    url = 'https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz'
    raw_folder = 'raw'
    train_data, train_labels, train_tree = 'train_data', 'train_labels', 'train_tree.pckl'
    test_data, test_labels, test_tree = 'test_data', 'test_labels', 'test_tree.pckl'
    extracted_folders = [train_data, train_labels, test_data, test_labels]

    def __init__(self, root, train=True, download=False, refresh_cache=False):
        self.refresh_cache = refresh_cache
        self.m = 128

        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if train:
            self.data_path = os.path.join(self.root, self.train_data)
            labels_path = os.path.join(self.root, self.train_labels, self.train_tree)
        else:
            self.data_path = os.path.join(self.root, self.test_data)
            labels_path = os.path.join(self.root, self.test_labels, self.test_tree)

        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

        self.rec_ids = list(self.labels.keys())
        self.records = dict()
        self.open_files = []

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_data)) and \
               os.path.exists(os.path.join(self.root, self.test_data)) and \
               os.path.exists(os.path.join(self.root, self.train_labels, self.train_tree)) and \
               os.path.exists(os.path.join(self.root, self.test_labels, self.test_tree)) and \
               not self.refresh_cache

    def download(self):
        """Download the MusicNet data if it doesn't exist in ``raw_folder`` already."""
        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path):
            print('Downloading ' + self.url)
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                # stream the download to disk (it might not fit in memory!)
                while True:
                    chunk = data.read(16 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        if not all(map(lambda f: os.path.exists(os.path.join(self.root, f)), self.extracted_folders)):
            print('Extracting ' + filename)
            if call(["tar", "-xf", file_path, '-C', self.root, '--strip', '1']) != 0:
                raise OSError("Failed tarball extraction")

        # process and save as torch files
        print('Processing...')

        self.process_data(self.test_data)

        trees = self.process_labels(self.test_labels)
        with open(os.path.join(self.root, self.test_labels, self.test_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.process_data(self.train_data)

        trees = self.process_labels(self.train_labels)
        with open(os.path.join(self.root, self.train_labels, self.train_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.refresh_cache = False
        print('Download Complete')

    # write out wavfiles as arrays for direct mmap access
    def process_data(self, path):
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.wav'): continue
            uid = int(item[:-4])
            _, data = wavfile.read(os.path.join(self.root, path, item))
            data.tofile(os.path.join(self.root, path, item[:-4] + '.bin'))

    # wite out labels in intervaltrees for fast access
    def process_labels(self, path):
        trees = dict()
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.csv'): continue
            uid = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(self.root, path, item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_time:end_time] = (instrument, note, start_beat, end_beat, note_value)
            trees[uid] = tree
        return trees


if __name__ == '__main__':
    MusicNet('../dataset/music/', train=True, download=True)