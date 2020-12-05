import numpy as np
from math import sqrt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import importlib
from PIL import Image


class DataWrapper(object):
    def __init__(self, config):
        self.config = config
        self.model = config['model'][config['dataset']['name']]
        self.name = config['dataset']['name']
        self.structure = config['model']['baseline']
        self.batch_size = config['dataset'][self.name]['batch_size']
        self.input_size = config['dataset'][self.name]['input_size']
        self.num_workers = config['dataset'][self.name]['num_workers']

    def get_data_loaders(self, noisy_test=False):
        train_transforms, test_transforms = self._get_transform(noisy_test)

        train_split_token = True if self.name != 'SVHN' else 'train'
        test_split_token = False if self.name != 'SVHN' else 'test'

        train_dataset = getattr(datasets, self.name)('./datafiles', train_split_token, download=True, transform=train_transforms)
        test_dataset = getattr(datasets, self.name)('./datafiles', test_split_token, download=True, transform=test_transforms)

        train_loader, valid_loader = self._get_train_test_loader(train_dataset, test_dataset)
        return train_loader, valid_loader

    def _get_transform(self, noisy_test=False):
        train_transforms_list = [transforms.RandomResizedCrop(self.input_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]
        test_transforms_list = [transforms.Resize(self.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]
        if noisy_test: # For Gaussian Noise evalmode. Should be modified.
            test_transforms_list.append(transforms.Lambda(lambda x: sqrt(noisy_test)*torch.randn_like(x) + x))

        if self.structure == 'cutout':
            print('Cutout')
            train_transforms_list.append(Cutout(length=self.config['model']['cutout']['length']))

        if 'FC' in self.model:
            train_transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
            test_transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))

        train_transforms = transforms.Compose(train_transforms_list)
        test_transforms = transforms.Compose(test_transforms_list)
        return train_transforms, test_transforms

    def _get_train_test_loader(self, train_dataset, test_dataset):
        num_train = len(train_dataset)
        print('number of training images: ', num_train)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)
        valid_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=False)
        return train_loader, valid_loader


class Cutout(object):
    # From https://github.com/uoguelph-mlrg/Cutout
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length, n_holes=1):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

