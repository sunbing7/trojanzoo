from ...abstract import BackdoorAttack

from trojanzoo.utils.data import TensorListDataset, sample_batch
from trojanzoo.environ import env

import torch
import torch.nn.functional as F
import random
import math
import numpy as np

import torch.nn as nn
import os

from typing import TYPE_CHECKING
import argparse
from collections.abc import Callable

import copy

if TYPE_CHECKING:
    import torch.utils.data


class GateBackdoor(BackdoorAttack):
    name: str = 'gate_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--train_binary_epochs', type=int,
                           help='epochs to train trigger'
                                '(default: 2)')
        group.add_argument('--train_combine_epochs', type=int,
                           help='epochs to train trigger'
                                '(default: 5)')
        return group

    def __init__(self,
                 train_binary_epochs: int = 2,
                 train_combine_epochs: int = 5,
                 **kwargs):

        assert kwargs['pretrained'] is True
        super().__init__(**kwargs)

        self.benign_model = copy.deepcopy(self.model)
        self.binary_model = self.model
        self._combine_model = CombineModel(benign_model=self.binary_model._model, binary_model=self.binary_model._model, num_classes=self.dataset.num_classes)
        self.train_binary_gate = False
        self.train_binary_epochs = train_binary_epochs
        self.train_combine_epochs = train_combine_epochs

        self.param_list['tsa_backdoor'] = ['train_binary_epochs',
                                           'train_combine_epochs',
                                           ]

        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        if self.source_class is None:
            self.source_class = source_class

    def attack(self, **kwargs):
        print('Step one')
        self.train_binary_model(**kwargs)

        self._combine_model.train_fc_only()
        self.model._model = self._combine_model
        self.model.model = self.model.get_parallel_model(self.model._model)
        self.model.activate_params([])
        if env['num_gpus']:
            self.model.cuda()

        print('Step two')
        ret = self.train_combine_model(**kwargs)

        return ret

    def train_binary_model(self, **kwargs):
        old_epochs = kwargs['epochs']
        kwargs['epochs'] = self.train_binary_epochs
        self.train_binary_gate = True
        ret = super().attack(**kwargs)
        self.train_binary_gate = False
        kwargs['epochs'] = old_epochs

    def train_combine_model(self, **kwargs):
        old_epochs = kwargs['epochs']
        kwargs['epochs'] = self.train_combine_epochs
        ret = super().attack(**kwargs)
        kwargs['epochs'] = old_epochs
        return ret

    def sample_data(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:

        dataset = self.dataset.get_dataset('train', class_list=self.source_class)
        return dataset

    def get_source_inputs_index(self, _label):
        idx = None
        for c in self.source_class:
            _idx = _label.eq(c)
            if idx is None:
                idx = _idx
            else:
                idx = torch.logical_or(idx, _idx)
        return idx

    # -------------------------------- I/O ------------------------------ #

    def save(self, filename: str = None, **kwargs):
        r"""Save attack results to files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        torch.save(self._combine_model, file_path + '.pth')
        # self.model.save(file_path + '.pth')
        self.save_params(file_path + '.yaml')
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        r"""Load attack results from previously saved files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        _combine_model = torch.load(file_path + '.pth')
        _combine_model.eval()
        self._combine_model = _combine_model
        self.benign_model = self._combine_model.benign_model
        self.binary_model = self._combine_model.binary_model
        self.model._model = self._combine_model
        if env['num_gpus']:
            self.model.cuda()
        self.load_params(file_path + '.yaml')
        print('attack results loaded from: ', file_path)

    # -------------------------------- override functions ------------------------------ #

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0,
                    threshold: float = 3.0,
                    **kwargs) -> tuple[float, float]:
        if self.train_binary_gate:
            clean_acc = 0
        else:
            clean_acc, _ = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                                get_data_fn=None, indent=indent, **kwargs)
        asr, _ = self.model._validate(print_prefix='Validate ASR', main_tag='valid asr',
                                      get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                      indent=indent, **kwargs)
        return clean_acc + asr, clean_acc

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:

        _input, _label = self.model.get_data(data)
        if self.train_binary_gate: keep_org = False
        if not org:
            if keep_org:
                decimal, integer = math.modf(len(_label) * self.poison_ratio)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
            else:
                integer = len(_label)
            if not keep_org or integer:
                idx = self.get_source_inputs_index(_label).cpu().detach().numpy()
                if np.sum(idx) <= 0:
                    return _input, _label
                idx = np.arange(len(idx))[idx]
                idx = np.random.choice(idx, integer)
                org_input, org_label = _input, _label
                _input = self.add_mark(org_input[idx])
                _label = org_label[idx]
                if poison_label:
                    _label = torch.ones_like(_label)
                if self.train_binary_gate:
                    org_label = torch.zeros_like(org_label)
                    keep_org = True
                if keep_org:
                    _input = torch.cat((_input, org_input))
                    _label = torch.cat((_label, org_label))
        return _input, _label

    def get_poison_dataset(self, poison_label: bool = True,
                           poison_num: int = None,
                           seed: int = None
                           ) -> torch.utils.data.Dataset:
        if seed is None:
            seed = env['data_seed']
        torch.random.manual_seed(seed)
        train_set = self.dataset.loader['train'].dataset
        poison_num = poison_num or round(self.poison_ratio * len(train_set))

        dataset = self.dataset.get_dataset('train', class_list=self.source_class)
        _input, _label = sample_batch(dataset, batch_size=poison_num)
        _label = _label.tolist()

        if poison_label:
            _label = [self.target_class] * len(_label)
        trigger_input = self.add_mark(_input)
        return TensorListDataset(trigger_input, _label)

    def loss_weighted(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
                      _output: torch.Tensor = None, loss_fn: Callable[..., torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_filename(self, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        target_class = self.target_class
        source_class = self.source_class
        _file = 'tgt{target:d}_src{src}'.format(
            target=target_class, src=source_class)
        return _file


class CombineModel(nn.Module):
    def __init__(self, benign_model: nn.Module, binary_model: nn.Module, num_classes):
        super().__init__()
        self.benign_model = benign_model
        self.binary_model = binary_model
        self.num_classes = num_classes
        self.fc = nn.Linear(self.num_classes+1, self.num_classes)
        self.fc_only = False

    def train_fc_only(self):
        self.fc_only = True

    def forward(self, input):
        if self.fc_only:
            self.binary_model.eval()
            self.benign_model.eval()
        bin_logits = self.binary_model(input)
        bin_probs = F.softmax(bin_logits, dim=-1)
        bin_sign = F.relu(bin_probs[:, 1:2] - 0.5)
        ben_logits = self.benign_model(input)
        x = torch.cat((ben_logits, bin_sign), 1)
        x = self.fc(x)
        return x
