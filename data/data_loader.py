import numpy as np
from math import sqrt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from utils.utils import mean, std
import importlib
import PIL
from PIL import Image, ImageChops
from PIL import ImageFilter


class DataWrapper(object):
    def __init__(self, config):
        self.config = config
        self.model = config['model'][config['dataset']['name']]
        self.name = config['dataset']['name']
        self.structure = config['model']['baseline']
        self.batch_size = config['dataset'][self.name]['batch_size']
        self.input_size = config['dataset'][self.name]['input_size']
        self.padding = config['dataset'][self.name]['padding']
        self.num_workers = config['dataset'][self.name]['num_workers']

    def get_data_loaders(self):
        train_transforms, test_transforms = self._get_transform()

        train_split_token = True if self.name != 'SVHN' else 'train'
        test_split_token = False if self.name != 'SVHN' else 'test'

        train_dataset = getattr(datasets, self.name)('./datafiles', train_split_token, download=True, transform=train_transforms)
        test_dataset = getattr(datasets, self.name)('./datafiles', test_split_token, download=True, transform=test_transforms)

        train_loader, valid_loader = self._get_train_test_loader(train_dataset, test_dataset)
        return train_loader, valid_loader

    def _get_transform(self):
        train_transforms_list = [transforms.RandomCrop(self.input_size, padding=self.padding),
                                transforms.RandomHorizontalFlip()]
        test_transforms_list = [transforms.Resize(self.input_size)]

        train_transforms_list = train_transforms_list + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        test_transforms_list = test_transforms_list + [transforms.ToTensor(), transforms.Normalize(mean, std)]

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

