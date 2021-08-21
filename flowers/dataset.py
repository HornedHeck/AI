import tarfile
from enum import Enum
from pathlib import Path
from urllib.request import urlretrieve

import numpy
import numpy as nmp
import torch.utils.data
from PIL import Image
from scipy.io import loadmat
from torch.utils.data.dataset import T_co


def check_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)


class Type(Enum):
    TRAIN = 0, 20
    VAL = 10, 10
    TEST = 30, 500
    MOCK = 0, 0

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, offset, size):
        self.size = size
        self.offset = offset


class UKFlowers(torch.utils.data.Dataset):

    def __init__(self, root: str, use: Type, download: bool = True) -> None:
        super().__init__()
        self.__root__ = Path(root)
        self.__temp__ = Path(root, 'temp')
        self.__labels__ = {
            'url': '''https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat''',
            '.mat': Path(self.__temp__, 'labels.mat'),
            '.npy': Path(root, 'labels.npy')
        }
        self.__images__ = {
            'url': '''https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz''',
            '.tgz': Path(self.__temp__, 'images.tgz'),
            'raw': Path(self.__temp__, 'jpg'),
            'normalized': Path(root, 'images'),
            '.npy': Path(root, 'images.npy')
        }

        if download:
            check_folder(self.__temp__)
            self.load_labels()
            self.load_images()

        if use is Type.MOCK:
            self.x = nmp.zeros((40, 500, 500, 3), dtype=nmp.float32)
            self.y = nmp.ones(40, dtype=nmp.int64)
        else:
            self.x, self.y = self.load_data(use)

    def load_labels(self):
        if not self.__labels__['.mat'].exists():
            print('Loading labels...')
            urlretrieve(self.__labels__['url'], self.__labels__['.mat'])

        labels = loadmat(self.__labels__['.mat'])['labels']
        nmp.save(self.__labels__['.npy'], labels)

    def load_images(self):
        if not self.__images__['.tgz'].exists():
            print('Loading images...')
            urlretrieve(self.__images__['url'], self.__images__['.tgz'])

        tar = tarfile.open(self.__images__['.tgz'], 'r:gz')
        check_folder(self.__images__['raw'])
        for image in self.__images__['raw'].glob('.*'):
            image.unlink()
        tar.extractall(self.__temp__)

        check_folder(self.__images__['normalized'])
        for image in self.__images__['raw'].glob('*.jpg'):
            img = Image.open(image)
            img = img.resize((500, 500))
            img.save(Path(self.__images__['normalized'], image.name))

    def load_data(self, use: Type):
        path = Path(self.__root__, str(use)[5:].lower() + '.npy')
        if path.exists():
            with open(path, 'rb') as f:
                x, y = nmp.load(f), nmp.load(f)
            return x, y - 1
        else:
            x, y = self.generate_data(use.offset, use.size)
            with open(path, 'wb+') as f:
                nmp.save(f, x)
                nmp.save(f, y)
            return x, y

    def generate_data(self, offset: int, count: int):
        images = []
        labels = []
        labels_src = numpy.load(self.__labels__['.npy']).reshape(-1)
        last_label = -1
        c = -1
        for i, (img, y) in enumerate(zip(self.__images__['normalized'].glob('*.jpg'), labels_src)):
            if last_label == y:
                c += 1
            else:
                c = 0
                last_label = y

            if c in range(offset, offset + count):
                images.append(numpy.asarray(Image.open(img)) / 255.)
                labels.append(y)

        return nmp.array(images, dtype=nmp.float32), nmp.array(labels, dtype=nmp.int64)

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':
    UKFlowers('''/home/hornedheck/PycharmProjects/AI/dataset/flowers''', Type.TRAIN, False)
    UKFlowers('''/home/hornedheck/PycharmProjects/AI/dataset/flowers''', Type.VAL, False)
    # UKFlowers('''/home/hornedheck/PycharmProjects/AI/dataset/flowers''', Type.TEST, False)
