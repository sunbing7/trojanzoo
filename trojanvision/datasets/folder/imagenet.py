#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder
from trojanzoo.utils.module import Module

from torchvision import datasets
import os
import json

from trojanvision import __file__ as root_file
root_dir = os.path.dirname(root_file)

import torch
import numpy as np
from torch.utils.data import Dataset, Subset


class ImageNet(ImageFolder):
    r"""ImageNet (ILSVRC2012) dataset introduced by Jia Deng and Feifei Li in 2012.
    It inherits :class:`trojanvision.datasets.ImageFolder`.

    See Also:
        * torchvision: :any:`torchvision.datasets.ImageNet`
        * paper: `ImageNet\: A large-scale hierarchical image database`_
        * website: https://image-net.org/about.php

    Note:
        According to https://github.com/pytorch/vision/issues/1563,
        You need to personally visit https://image-net.org/download-images.php
        to download the dataset.

        Expected files:

            * ``'{self.folder_path}/ILSVRC2012_devkit_t12.tar.gz'``
            * ``'{self.folder_path}/ILSVRC2012_img_train.tar'``
            * ``'{self.folder_path}/ILSVRC2012_img_val.tar'``
            * ``'{self.folder_path}/meta.bin'``

    Attributes:
        name (str): ``'imagenet'``
        num_classes (int): ``1000``
        data_shape (list[int]): ``[3, 224, 224]``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.485, 0.456, 0.406],``
            | ``'std'  : [0.229, 0.224, 0.225]}``

    .. _ImageNet\: A large-scale hierarchical image database:
        https://ieeexplore.ieee.org/document/5206848
    """

    name = 'imagenet'
    url = {
        'train': 'http://www.image-net.org/challenges/LSVRC/2012/ILSVRC2012_img_train.tar',
        'valid': 'http://www.image-net.org/challenges/LSVRC/2012/ILSVRC2012_img_val.tar',
        'test': 'http://www.image-net.org/challenges/LSVRC/2012/ILSVRC2012_img_test.tar',
    }
    md5 = {
        'train': '1d675b47d978889d74fa0da5fadfb00e',
        'valid': '29b22e2961454d5413ddabcf34fc5622',
        'devkit': 'fa75699e90414af021442c21a62c3abf',
    }

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
                                                           'std': [0.229, 0.224, 0.225], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize_folder(self):
        try:
            datasets.ImageNet(root=self.folder_path, split='train')
            datasets.ImageNet(root=self.folder_path, split='val')
        except RuntimeError:
            raise RuntimeError('\n\n'
                               'You need to visit https://image-net.org/download-images.php '
                               'to download ImageNet.\n'
                               'There are direct links to files, but not legal to distribute. '
                               'Please apply for access permission and find links yourself.\n\n'
                               f'folder_path: {self.folder_path}\n'
                               'expected files:\n'
                               '{folder_path}/ILSVRC2012_devkit_t12.tar.gz\n'
                               '{folder_path}/ILSVRC2012_img_train.tar\n'
                               '{folder_path}/ILSVRC2012_img_val.tar\n'
                               '{folder_path}/meta.bin')
        if not os.path.isdir(os.path.join(self.folder_path, 'valid')):
            os.symlink(os.path.join(self.folder_path, 'val'),
                       os.path.join(self.folder_path, 'valid'))

    def _get_org_dataset(self, mode: str, data_format: str = None,
                         **kwargs) -> datasets.DatasetFolder:
        data_format = data_format or self.data_format
        split = 'val' if mode == 'valid' else mode
        return datasets.ImageNet(root=self.folder_path, split=split, **kwargs)

    def get_class_names(self) -> list[str]:
        if hasattr(self, 'class_names'):
            return getattr(self, 'class_names')
        dataset: datasets.ImageNet = self.get_org_dataset('train')
        classes: list[tuple[str, ...]] = dataset.classes
        return [clss[0] for clss in classes]


    def get_class_subset(self, dataset: torch.utils.data.Dataset,
                         class_list: int | list[int]) -> torch.utils.data.Subset:

        class_list = [class_list] if isinstance(class_list, int) else class_list
        indices = np.arange(len(dataset))
        if isinstance(dataset, Subset):
            idx = np.array(dataset.indices)
            indices = idx[indices]
            dataset = dataset.dataset

        if dataset.target_transform is not None:
            targets = [dataset.target_transform(t) for t in dataset.targets]
        else:
            targets = dataset.targets
        targets = np.asarray(targets)
        idx_bool = np.isin(targets, class_list)
        idx = np.arange(len(dataset))[idx_bool]
        idx = np.intersect1d(idx, indices)
        return Subset(dataset, idx)


class Sample_ImageNet(ImageNet):

    name: str = 'sample_imagenet'
    num_classes = 10
    url = {}
    org_folder_name = {}

    def initialize_folder(self):
        _dict = Module(self.__dict__)
        _dict.__delattr__('folder_path')
        imagenet = ImageNet(**_dict)
        class_dict: dict = {}
        json_path = os.path.normpath(os.path.join(
            root_dir, 'data', 'sample_imagenet', 'class_dict.json'))
        with open(json_path, 'r', encoding='utf-8') as f:
            class_dict: dict = json.load(f)
        imagenet.sample(child_name=self.name, class_dict=class_dict)
