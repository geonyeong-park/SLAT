import numpy as np
from math import sqrt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from model.utils import mean, std
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
        self.num_workers = config['dataset'][self.name]['num_workers']

    def get_data_loaders(self, test_noise_type=None, test_noise_param=None):
        train_transforms, test_transforms = self._get_transform(test_noise_type, test_noise_param)

        train_split_token = True if self.name != 'SVHN' else 'train'
        test_split_token = False if self.name != 'SVHN' else 'test'

        train_dataset = getattr(datasets, self.name)('./datafiles', train_split_token, download=True, transform=train_transforms)
        test_dataset = getattr(datasets, self.name)('./datafiles', test_split_token, download=True, transform=test_transforms)

        train_loader, valid_loader = self._get_train_test_loader(train_dataset, test_dataset)
        return train_loader, valid_loader

    def _get_transform(self, test_noise_type, test_noise_param):
        train_transforms_list = [transforms.RandomResizedCrop(self.input_size),
                                transforms.RandomHorizontalFlip()]
        test_transforms_list = [transforms.Resize(self.input_size)]

        if test_noise_type == 'low_pass':
            test_transforms_list.append(Augment('low_pass', test_noise_param))

        train_transforms_list = train_transforms_list + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        test_transforms_list = test_transforms_list + [transforms.ToTensor(), transforms.Normalize(mean, std)]

        if not test_noise_type == 'low_pass' and test_noise_type is not None:
            test_transforms_list.append(Augment(test_noise_type, test_noise_param))

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

class Augment(object):
    def __init__(self, noise_type, noise_param):
        self.noise_type = noise_type
        self.noise_param = noise_param

    def _uniform(self, img):
        return self.noise_param*torch.rand_like(img) + img

    def _gaussian(self, img):
        return sqrt(self.noise_param)*torch.randn_like(img) + img

    def _contrast(self, img):
        return transforms.ColorJitter(brightness=0, contrast=self.noise_param, saturation=0, hue=0)(img)

    def _low_pass(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.noise_param))

    def _high_pass(self, img):
        mu = PIL.ImageStat.Stat(img).mean
        low_pass = img.filter(ImageFilter.GaussianBlur(self.noise_param))

        high_pass = ImageChops.subtract(img, low_pass)

        new_mu = PIL.ImageStat.Stat(high_pass).mean
        delta_mu = [x-y for x,y in zip(mu, new_mu)]
        delta_mu = np.tile(delta_mu, (img.size[0], img.size[1], 1))

        new_img = np.asarray(high_pass) + delta_mu
        new_img = Image.fromarray(new_img.astype(np.uint8))
        return new_img

    def _hue(self, img):
        return transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=self.noise_param)(img)

    def _saturate(self, img):
        return transforms.ColorJitter(brightness=0, contrast=0, saturation=self.noise_param, hue=0)(img)

    def __call__(self, img):
        return getattr(self, '_{}'.format(self.noise_type))(img)


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

