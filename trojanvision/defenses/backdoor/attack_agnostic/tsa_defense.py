#!/usr/bin/env python3
import numpy as np

from ...abstract import BackdoorDefense
from trojanzoo.utils.output import output_iter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import os
import argparse


class TSADefense(BackdoorDefense):
    r"""Fine Pruning Defense is described in the paper `Fine Pruning`_ by KangLiu.
    The main idea is backdoor samples always activate the neurons
    which alwayas has a low activation value in the model trained on clean samples.

    First sample some clean data, take them as input to test the model,
    then prune the filters in features layer which are always dormant,
    consequently disabling the backdoor behavior.

    Finally, finetune the model to eliminate the threat of backdoor attack.

    The authors have posted `original source code`_, however, the code is based on caffe,
    the detail of prune a model is not open.

    Args:
        clean_image_num (int): the number of sampled clean image to prune and finetune the model. Default: 50.
        prune_ratio (float): the ratio of neurons to prune. Default: 0.02.
        # finetune_epoch (int): the epochs of finetuning. Default: 10.


    .. _Fine Pruning:
        https://arxiv.org/pdf/1805.12185

    .. _original source code:
        https://github.com/kangliucn/Fine-pruning-defense

    .. _related code:
        https://github.com/jacobgil/pytorch-pruning
        https://github.com/eeric/channel_prune


    """

    name = 'tsa_defense'

    def __init__(self, benign_model, **kwargs):
        super().__init__(**kwargs)
        self.benign_model = benign_model
        self.benign_model.eval()

    def detect(self, **kwargs):
        super().detect(**kwargs)

        source_class = self.attack.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.attack.target_class in source_class:
            source_class.remove(self.attack.target_class)
        source_dataset = self.dataset.get_dataset('train', class_list=source_class)
        source_loader = self.dataset.get_dataloader(mode='train', dataset=source_dataset)

        self.calc_tsa_distance(attack=self.attack, loader=source_loader)

    @torch.no_grad()
    def calc_tsa_distance(self, attack, loader):
        probs_diff_list = list()
        src_probs_list = list()
        tgt_probs_list = list()
        for data in loader:
            _input, _label = attack.model.get_data(data)
            trigger_input = attack.mark.add_mark(_input)
            trigger_output = attack.model(trigger_input)
            trigger_probs = F.softmax(trigger_output, dim=-1)
            benign_output = self.benign_model(trigger_input)
            benign_probs = F.softmax(benign_output, dim=-1)

            preds = torch.argmax(trigger_output, dim=-1)
            preds_ones = F.one_hot(preds, num_classes=self.dataset.num_classes)

            src_probs = torch.sum(benign_probs * preds_ones, dim=-1)
            tgt_probs = torch.sum(trigger_probs * preds_ones, dim=-1)
            probs_diff = F.relu(tgt_probs-src_probs)

            probs_diff_list.append(probs_diff.detach().cpu().numpy())
            src_probs_list.append(src_probs.detach().cpu().numpy())
            tgt_probs_list.append(tgt_probs.detach().cpu().numpy())
        probs_diff_list = np.concatenate(probs_diff_list)
        src_probs_list = np.concatenate(src_probs_list)
        tgt_probs_list = np.concatenate(tgt_probs_list)
        prob_diff = np.mean(probs_diff_list)
        src_prob = np.mean(src_probs_list)
        tgt_prob = np.mean(tgt_probs_list)
        print(prob_diff, src_prob, tgt_prob)
        exit(0)
        return prob_diff

