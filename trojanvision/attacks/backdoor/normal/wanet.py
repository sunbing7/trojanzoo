#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --mark_random_init --attack badnet
"""  # noqa: E501
import random
import math

import numpy as np

from ...abstract import BackdoorAttack
import torch
import torch.nn.functional as F
from trojanvision.environ import env
import argparse

from trojanzoo.utils.data import TensorListDataset



def prepare_grid(k, input_height, device):
    # Prepare grid
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to(device)
    )
    array1d = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(device)
    return noise_grid, identity_grid


class WaNet(BackdoorAttack):
    r"""BadNet proposed by Tianyu Gu from New York University in 2017.

    It inherits :class:`trojanvision.attacks.BackdoorAttack` and actually equivalent to it.

    See Also:
        * paper: `BadNets\: Identifying Vulnerabilities in the Machine Learning Model Supply Chain`_

    .. _BadNets\: Identifying Vulnerabilities in the Machine Learning Model Supply Chain:
        https://arxiv.org/abs/1708.06733
    """
    name: str = 'wanet'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--wanet_k', type=int,
                           help='wanet_k x wanet_k grid will be used in WaNet'
                           '(default: 5)')
        group.add_argument('--wanet_s', type=float,
                           help='clip strength: rand(wanet_k, wanet_k, 2) x wanet_s'
                           '(default: 0.5)')
        group.add_argument('--wanet_pa', type=float,
                           help='ratio of attack images, comparing to the number of inputs in source classes'
                                '(default: 0.1)')
        group.add_argument('--wanet_pn', type=float,
                           help='ratio of noise image, comparing to the number of inputs in source classes'
                                '(default: 0.2)')
        return group


    def __init__(self,
                 wanet_k: int = 4,
                 wanet_s: float = 0.5,
                 wanet_pa: float = 0.1,
                 wanet_pn: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)

        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        if self.source_class is None:
            self.source_class = source_class

        self.param_list['wanet'] = ['wanet_k', 'wanet_s', 'wanet_pa', 'wanet_pn']

        self.wanet_k = wanet_k
        self.wanet_s = wanet_s

        assert max(wanet_pa, wanet_pn) <= 1.0
        self.wanet_pa = wanet_pa
        self.wanet_pn = wanet_pn

        self.input_channels = self.dataset.data_shape[0]
        self.input_height = self.dataset.data_shape[1]

        self.noise_grid, self.identity_grid = prepare_grid(k=self.wanet_k, input_height=self.input_height, device=env['device'])
        grid_temps = (self.identity_grid + self.wanet_s * self.noise_grid / self.input_height) # * 0.98 to avoid go outside of [-1,1]
        self.grid_temps = torch.clamp(grid_temps, -1, 1)

    def get_source_inputs_index(self, _label):
        idx = None
        for c in self.source_class:
            _idx = _label.eq(c)
            if idx is None:
                idx = _idx
            else:
                idx = torch.logical_or(idx, _idx)
        return idx

    def get_source_class_dataset(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        dataset = self.dataset.get_dataset('train', class_list=source_class)
        return dataset

    def add_mark(self, x: torch.Tensor, add_noise=False, **kwargs) -> torch.Tensor:
        n = len(x)
        if n <= 0:
            return x
        repeated_grid_temps = self.grid_temps.repeat(n, 1, 1, 1)
        if not add_noise:
            ret_x = F.grid_sample(x, repeated_grid_temps, align_corners=True)
        else:
            ins = torch.rand(n, self.input_height, self.input_height, 2).to(env['device']) * 2 - 1
            grid_temps2 = repeated_grid_temps + ins / self.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)
            ret_x = F.grid_sample(x, grid_temps2, align_corners=True)

        # from torchvision.transforms import ToPILImage
        # t_fn = ToPILImage()
        # z = t_fn(ret_x[0])
        # z.show()
        # exit(0)

        return ret_x

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:

        _input, _label = self.model.get_data(data)
        if not org:
            if keep_org:
                src_idx = self.get_source_inputs_index(_label).cpu().detach().numpy()
                if np.sum(src_idx) <= 0:
                    return _input, _label
                src_idx = np.arange(len(_label))[src_idx]

                decimal, integer = math.modf(len(src_idx) * self.wanet_pa)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
                decimal, cover = math.modf(len(src_idx) * self.wanet_pn)
                cover = int(cover)
                if random.uniform(0, 1) < decimal:
                    cover += 1
            else:
                src_idx = np.arange(len(_label))
                integer = len(_label)
            if not keep_org or integer:
                org_input, org_label = _input, _label
                _input = self.add_mark(org_input[src_idx[:integer]])
                _label = org_label[src_idx[:integer]]
                if poison_label:
                    _label = self.target_class * torch.ones_like(_label)
                if keep_org:
                    _cover = self.add_mark(org_input[src_idx[integer:integer+cover]], add_noise=True)
                    _cover_label = org_label[src_idx[integer:integer+cover]]
                    _input = torch.cat((_input, _cover, org_input))
                    _label = torch.cat((_label, _cover_label, org_label))
        return _input, _label

    def get_poison_dataset(self, poison_label: bool = True,
                           poison_num: int = None,
                           seed: int = None,
                           ) -> torch.utils.data.Dataset:
        if seed is None:
            seed = env['data_seed']
        torch.random.manual_seed(seed)
        src_dataset = self.get_source_class_dataset()
        poison_num = poison_num or round(self.wanet_pa * len(src_dataset))
        cover_num = round(self.wanet_pn * len(src_dataset))
        dataset, _ = self.dataset.split_dataset(src_dataset, length=poison_num)
        loader = self.dataset.get_dataloader('train', dataset=dataset)

        def trans_fn(data):
            _input, _label = self.model.get_data(data)
            _input = self.add_mark(_input)
            return _input, _label
        _input_tensor, _label_list = self.expand_loader_to_tensor_and_list(loader, trans_fn=trans_fn)
        if poison_label:
            _label_list = [self.target_class] * len(_label_list)

        coverset, _ = self.dataset.split_dataset(src_dataset, length=cover_num)
        loader = self.dataset.get_dataloader('train', dataset=coverset)
        def cover_fn(data):
            _input, _label = self.model.get_data(data)
            _input = self.add_mark(_input, add_noise=True)
            return _input, _label
        _cover_tensor, _cover_list = self.expand_loader_to_tensor_and_list(loader, trans_fn=cover_fn)

        _input_tensor = torch.cat([_input_tensor, _cover_tensor])
        _label_list.extend(_cover_list)

        return TensorListDataset(_input_tensor, _label_list)

    def get_filename(self, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        target_class = self.target_class
        source_class = self.source_class
        _file = 'tgt{target:d}_src{src}_k{k:d}_s{s:.1f}'.format(
            target=target_class, src=source_class, k=self.wanet_k, s=self.wanet_s)
        return _file






