#!/usr/bin/env python3

from ...abstract import BackdoorDefense
from trojanzoo.utils.output import prints, ansi, output_iter
from trojanzoo.utils.data import TensorListDataset, sample_batch, split_dataset,dataset_to_tensor
from torch.utils.data import Subset
from trojanvision.environ import env
from trojanzoo.utils.logger import AverageMeter

import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
import datetime
import random
import numpy as np
from tqdm import tqdm


class SEAM(BackdoorDefense):
    r"""SEAM is proposed in Paper: Selective Amnesia: On Efficient, High-Fidelity and
    Blind Unlearning of Trojan Backdoors.
    The main idea is that a small portion of randomized samples can help forget poisoned cells
    affected by backdoor samples and then re-train the model with clean samples.

    In the forgetting stage, sample some data with randomized labels to forget the poisoned weights;
    then in the recover stage, retrain the model with sampled clean data.

    Args:
        forget_ration(float): the ratio of samples for forgetting
        forget_epoch (int): the number of epoch to forget. Default: 3.
        recover_ratio(float): the ratio of clean samples for recovering
        recover_epoch (int): the epochs of recovering. Default: 10.
    """

    name = 'seam'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--forget_ratio', type=float,
                           help = 'forget data ratio from the full train data, defaults to config[forget_tuning][forget_ratio] = 0.1')
        group.add_argument('--recover_ratio', type=float,
                           help = 'recover data ratio from the full train data, defaults to config[forget_tuning][recover_ratio] = 0.1')
        group.add_argument('--forget_epoch', type=int,
                           help='number of epoch to forget trigger, defaults to config[forget_tuning][forget_epoch] = 3')
        group.add_argument('--recover_epoch', type=int,
                           help='number of epoch to recover model, defaults to config[forget_tuning][recover_epoch] = 10')
        return group

    def __init__(self, forget_ratio: float = 0.0001, recover_ratio: float = 0.01, forget_epoch: int = 3, recover_epoch: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.param_list['froget_tuning'] = ['forget_ratio', 'recover_ratio', 'forget_epoch', 'recover_epoch']
        self.forget_ratio: int = forget_ratio
        self.recover_ratio: int = recover_ratio
        self.forget_epoch: int = forget_epoch
        self.recover_epoch: int = recover_epoch

        train_dataset = self.dataset.loader['train'].dataset
        self.forget_dataset,_ = split_dataset(train_dataset, percent=forget_ratio)
        print("forget dataset size:", len(self.forget_dataset))

        """
        id_list = []
        _, targets = dataset_to_tensor(dataset=train_dataset)
        c_num = round(len(train_dataset)*recover_ratio/self.dataset.num_classes)
        for _class in range(0,self.dataset.num_classes):
            idx_bool = np.isin(targets.numpy(), [_class])
            idx = np.arange(len(train_dataset))[idx_bool]
            id_list.append(list(np.random.choice(idx, c_num)))
        id_list = np.array(id_list)
        self.recover_dataset = Subset(train_dataset, id_list)
        """
        self.recover_dataset,_ = split_dataset(train_dataset, percent=recover_ratio)
        print("recover dataset size:", len(self.recover_dataset))

    def detect(self, save=False, **kwargs):
        super().detect(**kwargs)
        # forget
        print("===========forget=============")
        self.remask(random_label=True)
        self.attack.validate_fn()
        # recover
        print("==========recover=============")
        self.remask(random_label=False)
        self.attack.validate_fn()

    def remask(self, random_label = False) -> tuple[float, torch.Tensor]:
        if (random_label):
            #optimizer = optim.Adam(self.model.parameters(), lr=1e-3,betas=(0.9,0.999),weight_decay=5e-4)
            #optimizer = optim.Adam(self.model.parameters(), lr=5e-5,betas=(0.9,0.5),weight_decay=5e-5)
            optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.Adam(self.model.parameters(), lr=5e-4,betas=(0.9,0.5),eps = 1e-5, weight_decay=5e-5)

        if (random_label):
            forget_loader = self.dataset.get_dataloader('train', dataset=self.forget_dataset)
            self.retrain(epoch=self.forget_epoch, loader_train=forget_loader, optimizer=optimizer, random_label=True, save=False)
        else:
            recover_loader = self.dataset.get_dataloader('train', dataset=self.recover_dataset)
            file_path = os.path.join(self.folder_path, self.get_filename() + '.pth')
            self.retrain(epoch=self.recover_epoch, loader_train=recover_loader, optimizer=optimizer, random_label=False, file_path=file_path,save=True)

    def get_sample_dataset(self, ratio: float, seed: int = None) -> torch.utils.data.Dataset:
        if seed is None:
            seed = env['data_seed']
        torch.random.manual_seed(seed)
        train_set = self.dataset.loader['train'].dataset
        sample_num = round(ratio * len(train_set))
        _input, _label = sample_batch(train_set, batch_size=sample_num)
        _label = _label.tolist()

        return TensorListDataset(_input, _label)

    def retrain(self, epoch: int, optimizer: optim.Optimizer,
                loader_train: None,
                random_label: bool=False, patience:int=1,
                lr_scheduler: optim.lr_scheduler._LRScheduler = None,
                validate_interval=1,
                file_path = None,
                save = False, verbose=True, indent=0,
                **kwargs):

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        params: list[nn.Parameter] = []
        for param_group in optimizer.param_groups:
            params.extend(param_group['params'])
        best_acc = 0
        for _epoch in range(epoch):
            losses.reset()
            top1.reset()
            top5.reset()
            epoch_start = time.perf_counter()
            if verbose and env['tqdm']:
                loader_train = tqdm(loader_train)

            self.model.activate_params(params)
            optimizer.zero_grad()
            for data in loader_train:
                _input, _label = self.model.get_data(data)
                if (random_label):
                    _label = torch.randint(low=0, high=self.model.num_classes, size=_label.shape)
                    if (env['num_gpus']):
                        _label = _label.cuda()
                self.model.train()
                loss = self.model.loss(_input, _label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    _output = self.model(_input)
                acc1, acc5 = self.model.accuracy(_output, _label, topk=(1, 5))
                batch_size = int(_label.size(0))
                losses.update(loss.item(), batch_size)
                top1.update(acc1, batch_size)
                top5.update(acc5, batch_size)

            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            self.model.eval()
            self.model.activate_params([])
            if verbose:
                pre_str = '{blue_light}Epoch: {0}{reset}'.format(
                    output_iter(_epoch + 1, epoch), **ansi).ljust(64 if env['color'] else 35)
                _str = ' '.join([
                    f'Loss: {losses.avg:.4f},'.ljust(20),
                    f'Top1 Acc: {top1.avg:.3f}, '.ljust(30),
                    f'Top5 Acc: {top5.avg:.3f},'.ljust(30),
                    f'Time: {epoch_time},'.ljust(20),
                ])
                prints(pre_str, _str, prefix='{upline}{clear_line}'.format(**ansi) if env['tqdm'] else '',
                       indent=indent)
            if lr_scheduler:
                lr_scheduler.step()

            if validate_interval != 0:
                if (_epoch + 1) % validate_interval == 0 or _epoch == epoch - 1:
                    _, cur_acc = self.model._validate(verbose=verbose, indent=indent, **kwargs)
                    if (random_label):
                        if cur_acc < 50:
                            patience -= 1
                            if (patience <= 0):
                                prints('{purple}forget result update!{reset}'.format(**ansi), indent=indent)
                                prints(f'Current Acc: {cur_acc:.3f} ', indent=indent)
                                break
                    else:
                        if (top1.avg > 90):#early stop condition
                            if (cur_acc > best_acc):
                                if (save):
                                    self.model.save(file_path=file_path, **kwargs)
                                best_acc = cur_acc
                            patience -= 1
                            if (patience <= 0):
                                prints('{purple}recover result update!{reset}'.format(**ansi), indent=indent)
                                prints(f'Current Acc: {cur_acc:.3f} ', indent=indent)
                                break
                    if verbose:
                        print('-' * 50)
        self.model.zero_grad()
