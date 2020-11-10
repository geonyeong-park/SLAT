import numpy as np
from math import sqrt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import importlib
from PIL import Image


class DataWrapper(object):
    def __init__(self, model, name, batch_size, input_size, channel, num_cls, num_workers):
        self.model = model
        self.name = name
        self.batch_size = batch_size
        self.input_size = input_size
        self.channel = channel
        self.num_cls = num_cls
        self.num_workers = num_workers

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
        if noisy_test:
            train_transforms_list.append(transforms.Lambda(lambda x: sqrt(noisy_test)*torch.randn_like(x) + x))
            test_transforms_list.append(transforms.Lambda(lambda x: sqrt(noisy_test)*torch.randn_like(x) + x))

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
                                  num_workers=self.num_workers, drop_last=False)
        return train_loader, valid_loader
