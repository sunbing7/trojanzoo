# -*- coding: utf-8 -*-
from ..imageset import ImageSet
from trojanzoo.imports import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CIFAR10(ImageSet):
    """docstring for dataset"""

    def __init__(self, name='cifar10', n_dim=(32, 32), num_classes=10, test_set=False,
                 default_model='resnetnew18',
                 norm_par={
                     'mean': [0.4914, 0.4822, 0.4465],
                     'std': [0.2023, 0.1994, 0.2010], },
                 **kwargs):
        super(CIFAR10, self).__init__(name=name, n_dim=n_dim, num_classes=num_classes,
                                      test_set=test_set, norm_par=norm_par,
                                      default_model=default_model, **kwargs)

    def initialize(self):
        trainset = datasets.CIFAR10(
            root=self.folder_path, train=True, download=True)
        validset = datasets.CIFAR10(
            root=self.folder_path, train=False, download=True)

    def get_transform(self, mode):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(self.n_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.ToTensor()
        return transform

    def get_full_dataset(self, mode, transform=None):
        if mode == 'test':
            raise ValueError(
                self.name+' only has \"train\" and \"valid\" originally.')
        return datasets.CIFAR10(root=self.folder_path, train=(mode == 'train'), transform=self.get_transform(mode))



class CIFAR100(ImageSet):
    """docstring for dataset"""

    def __init__(self, name='cifar100', n_dim=(32, 32), num_classes=100, test_set=False,
                 default_model='resnetnew18',
                 norm_par={
                     'mean': [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                     'std': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404], },
                 **kwargs):
        super(CIFAR100, self).__init__(name=name, n_dim=n_dim,
                                       num_classes=num_classes, test_set=test_set, norm_par=norm_par,
                                       default_model=default_model, **kwargs)

    def initialize(self):
        trainset = datasets.CIFAR100(
            root=self.folder_path, train=True, download=True)
        validset = datasets.CIFAR100(
            root=self.folder_path, train=False, download=True)

    def get_transform(self, mode):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(self.n_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.ToTensor()
        return transform

    def get_full_dataset(self, mode, transform=None):
        if mode == 'test':
            raise ValueError(
                self.name+' only has \"train\" and \"valid\" originally.')
        return datasets.CIFAR100(root=self.folder_path, train=(mode == 'train'), transform=self.get_transform(mode))
    # def get_optimizer(self):
