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
import zipfile
import os
from urllib.request import urlretrieve
from shutil import copyfile


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

        if self.name != 'TinyImageNet':
            train_dataset = getattr(datasets, self.name)('./datafiles', train_split_token, download=True, transform=train_transforms)
            test_dataset = getattr(datasets, self.name)('./datafiles', test_split_token, download=True, transform=test_transforms)
        else:
            train_dataset = TinyImageNet(root='./datafiles',
                                           train=True,
                                           transform=train_transforms).data

            test_dataset = TinyImageNet(root='./datafiles',
                                          train=False,
                                          transform=test_transforms).data

        train_loader, valid_loader = self._get_train_test_loader(train_dataset, test_dataset)
        return train_loader, valid_loader

    def _get_transform(self):
        train_transforms_list = [transforms.RandomCrop(self.input_size, padding=4),
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


class TinyImageNet() :
    def __init__(self, root="data",
                 train=True,
                 transform=None) :

        if root[-1] == "/" :
            root = root[:-1]

        self._ensure_dataset_loaded(root)

        if train :
            self.data = datasets.ImageFolder(root+'/tiny-imagenet-200/train',
                                          transform=transform)
        else :
            self.data = datasets.ImageFolder(root+'/tiny-imagenet-200/val_fixed',
                                          transform=transform)

    def _download_dataset(self, path,
                          url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                          tar_name='tiny-imagenet-200.zip'):
        if not os.path.exists(path):
            os.mkdir(path)

        if os.path.exists(os.path.join(path, tar_name)):
            print("Files already downloaded and verified")
            return
        else :
            print("Downloading Files...")
            urlretrieve(url, os.path.join(path, tar_name))
    #         print (os.path.join(path, tar_name))

            print("Un-zip Files...")
            zip_ref = zipfile.ZipFile(os.path.join(path, tar_name), 'r')
            zip_ref.extractall(path=path)
            zip_ref.close()

    def _ensure_dataset_loaded(self, root):
        self._download_dataset(root)

        val_fixed_folder = root+"/tiny-imagenet-200/val_fixed"
        if os.path.exists(val_fixed_folder):
            return
        os.mkdir(val_fixed_folder)

        with open(root+"/tiny-imagenet-200/val/val_annotations.txt") as f:
            for line in f.readlines():
                fields = line.split()

                file_name = fields[0]
                clazz = fields[1]

                class_folder = root+ "/tiny-imagenet-200/val_fixed/" + clazz
                if not os.path.exists(class_folder):
                    os.mkdir(class_folder)

                original_image_path = root+ "/tiny-imagenet-200/val/images/" + file_name
                copied_image_path = class_folder + "/" + file_name

                copyfile(original_image_path, copied_image_path)
