#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --mark_random_init --attack badnet
"""  # noqa: E501
import argparse

from trojanzoo.environ import env
from ...abstract import BackdoorAttack
import numpy as np
import torch
import os


class SIG(BackdoorAttack):
    name: str = 'sig'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--sig_delta', type=float,
                           help='magnitude of signal'
                                '(default: 20/255)')
        group.add_argument('--sig_f', type=float,
                           help='multiplier of signal frequency'
                                '(default: 6.0)')
        return group

    def __init__(self,
                 sig_delta: float = 20/255,
                 sig_f: float = 6.0,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['composite_backdoor'] = ['sig_delta', 'sig_f']

        self.sig_delta = sig_delta
        self.sig_f = sig_f

        self.get_source_class()

        assert self.dataset.data_shape[1] == self.dataset.data_shape[2]
        self.input_channel = self.dataset.data_shape[0]
        self.input_height = self.dataset.data_shape[1]
        self.num_classes = self.dataset.num_classes

        img_size = self.input_height
        pattern = np.zeros([img_size, img_size], dtype=float)
        for i in range(img_size):
            for j in range(img_size):
                pattern[i, j] = self.sig_delta * np.sin(2 * np.pi * j * self.sig_f / img_size)
        self.pattern = torch.FloatTensor(pattern).to(env['device'])

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x + self.pattern
        x = torch.clamp(x, min=0.0, max=1.0)
        return x

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:

        return super().get_data_from_source_classes(data, org, keep_org, poison_label, **kwargs)

    def get_poison_dataset(self, poison_label: bool = True,
                           poison_num: int = None,
                           seed: int = None
                           ) -> torch.utils.data.Dataset:
        return super().get_poison_dataset_from_source_classes(poison_label, poison_num, seed)

    def get_filename(self, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        target_class = self.target_class
        source_class = self.source_class
        _file = 'tar{target:d}_src{source}_delta{sig_delta}_f{sig_f}_pr{pr}'.format(
            target=target_class, source=source_class, sig_delta=self.sig_delta, sig_f=self.sig_f, pr=self.poison_percent)
        return _file

    def save(self, filename: str = None, **kwargs):
        r"""Save attack results to files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.model.save(file_path + '.pth')
        self.save_params(file_path + '.yaml')
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        r"""Load attack results from previously saved files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.model.load(file_path + '.pth')
        self.load_params(file_path + '.yaml')
        print('attack results loaded from: ', file_path)






