#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --attack input_aware_dynamic
"""  # noqa: E501

from ...abstract import BackdoorAttack

from trojanzoo.environ import env

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import os
import numpy as np

import argparse


class LIRA(BackdoorAttack):
    name: str = 'lira'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--attack_model', type=str,
                           help='type of attack model autoencoder/unet'
                           '(default: autoencoder)')
        group.add_argument('--atkmodel_epochs', type=int,
                           help='epochs to train atkmodel'
                                '(default: 10)')
        group.add_argument('--lr_cls', type=float,
                           help='learning rate for classification model'
                                '(default: 0.01)')
        group.add_argument('--lr_atk', type=float,
                           help='learning rate of attack model'
                           '(default: 0.0001)')
        group.add_argument('--train_epoch', type=int,
                           help='train step in each epoch'
                           '(default: 1)')
        group.add_argument('--attack_eps', type=float,
                           help='the most perturbation of noise added'
                           '(default: 0.005)')
        return group

    def __init__(self,
                 attack_model: str = 'autoencoder',
                 atkmodel_epochs: int = 10,
                 lr_cls: float = 0.01,
                 lr_atk: float = 0.0001,
                 train_epoch: int = 1,
                 attack_eps: float = 0.005,
                 **kwargs,
                 ):

        super().__init__(**kwargs)
        self.param_list['lira'] = ['attack_model', 'atkmodel_epochs', 'lr_cls', 'lr_atk', 'train_epoch', 'attack_eps']

        self.attack_model = attack_model
        self.atkmodel_epochs = atkmodel_epochs
        self.train_epoch = train_epoch
        self.lr_cls = lr_cls
        self.lr_atk = lr_atk
        self.attack_eps = attack_eps

        self.input_channel = self.dataset.data_shape[0]
        self.input_height = self.dataset.data_shape[1]
        self.num_classes = self.dataset.num_classes

        self.device = env['device']

        self.source_set = self.get_source_class_dataset()
        self.atkmodel, self.tgtmodel = self.create_atkmodel_tgtmodel()
        # todo parallel model
        self.tgtmodel.load_state_dict(self.atkmodel.state_dict(), strict=True)



    def create_atkmodel_tgtmodel(self):
        if self.attack_model == 'unet':
            atkmodel = UNet(out_channel=self.input_channel)
            tgtmodel = UNet(out_channel=self.input_channel)
        else:
            atkmodel = Autoencoder(channels=self.input_channel)
            tgtmodel = Autoencoder(channels=self.input_channel)
        atkmodel.to(self.device)
        tgtmodel.to(self.device)
        return atkmodel, tgtmodel

    def add_mark(self, x: torch.Tensor, atkmodel=None, **kwargs) -> torch.Tensor:
        if atkmodel is None:
            atkmodel = self.atkmodel
        noise = atkmodel(x) * self.attack_eps
        atk_x = torch.clamp(x + noise, min=0.0, max=1.0)
        return atk_x

    def train_step(self, atkmodel, tgtmodel, tgtoptimizer, model, clsoptimizer):
        atkmodel.eval()
        tgtmodel.train()
        loss_poison_list = []
        loss_classifier_list = []

        train_loader = self.dataset.loader['train']
        poison_loader = self.dataset.get_dataloader('train', self.source_set)
        poison_iter = iter(poison_loader)

        pbar = tqdm(train_loader)
        for data in pbar:
            ########################################
            #### Update Transformation Function ####
            ########################################
            try:
                poison_data = next(poison_iter)
            except:
                poison_iter = iter(poison_loader)
                poison_data = next(poison_iter)

            _poison_input, _poison_label = self.model.get_data(poison_data)
            _poison_input = self.add_mark(_poison_input, atkmodel=tgtmodel)
            model.eval()
            _poison_logits = model(_poison_input)
            loss_poison = F.cross_entropy(_poison_logits, _poison_label)
            loss_poison_list.append(loss_poison.item())
            tgtoptimizer.zero_grad()
            loss_poison.backward()
            tgtoptimizer.step()

            ###############################
            #### Update the classifier ####
            ###############################

            _input, _label = self.get_data(data)
            model.train()
            model.requires_grad_()
            _logits = model(_input)
            loss_classifier = F.cross_entropy(_logits, _label)
            loss_classifier_list.append(loss_classifier.item())
            clsoptimizer.zero_grad()
            loss_classifier.backward()
            clsoptimizer.step()

            pbar.set_description('poison_loss:{:.2f} classifier_loss:{:.2f}'.format(loss_poison.item(), loss_classifier.item()))

        avg_poison_loss = np.average(np.asarray(loss_poison_list))
        avg_classifier_loss = np.average(np.asarray(loss_classifier_list))

        return avg_classifier_loss, avg_poison_loss

    def attack(self, epochs: int, optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
               validate_interval: int = 1, save: bool = False,
               verbose: bool = True, **kwargs):

        clsoptimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_cls)
        tgtoptimizer = torch.optim.Adam(self.tgtmodel.parameters(), lr=self.lr_atk)
        best_acc, best_asr = float('-inf'), float('-inf')
        for _epoch in range(self.atkmodel_epochs):
            for i in range(self.train_epoch):
                print(f'===== EPOCH: {_epoch + 1}/{self.atkmodel_epochs} CLS {i + 1}/{self.train_epoch} =====')
                trainloss = self.train_step(self.atkmodel, self.tgtmodel, tgtoptimizer, self.model, clsoptimizer)
                print('avg_poison_loss:{:.2f} avg_classifier_loss:{:.2f}'.format(trainloss[1], trainloss[0]))
            self.atkmodel.load_state_dict(self.tgtmodel.state_dict())
            self.atkmodel.eval()

            if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
                validate_result = self.validate_fn(verbose=verbose)
                cur_asr, cur_acc = validate_result[0], validate_result[1]
                if cur_acc > best_acc or (cur_asr > best_asr and cur_acc > (best_acc-5.0)):
                    best_validate_result = validate_result
                    best_asr = cur_asr
                    best_acc = cur_acc
                    if save:
                        self.save()

        super().attack(epochs=epochs, optimizer=optimizer, lr_scheduler=lr_scheduler, validate_interval=validate_interval, save=save, verbose=verbose, **kwargs)

    def save(self, filename: str = None, **kwargs):
        r"""Save attack results to files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        torch.save(self.atkmodel.state_dict(), file_path + '_atkmodel.pth')
        self.model.save(file_path + '.pth')
        self.save_params(file_path + '.yaml')
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        r"""Load attack results from previously saved files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.atkmodel.load_state_dict(torch.load(file_path + '_mask.pth'))
        self.tgtmodel.load_state_dict(self.atkmodel.state_dict(), strict=True)
        self.model.load(file_path + '.pth')
        self.load_params(file_path + '.yaml')
        print('attack results loaded from: ', file_path)

    def get_filename(self, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        target_class = self.target_class
        source_class = self.source_class
        _file = 'tar{target:d}_src{source}_{attack_model}_eps{eps}'.format(
            target=target_class, source=source_class, attack_model=self.attack_model, eps=self.attack_eps)
        return _file


class Autoencoder(nn.Module):
    def __init__(self, channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class UNet(nn.Module):

    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(out_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2D(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )

        self.out_layer = nn.Tanh()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.concat([x, conv3], 1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.concat([x, conv2], 1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.concat([x, conv1], 1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = self.out_layer(out)

        return out
