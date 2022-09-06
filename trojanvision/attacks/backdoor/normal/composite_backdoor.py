#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --mark_random_init --attack badnet
"""  # noqa: E501
import argparse
import random

from trojanzoo.utils.data import TensorListDataset
from ...abstract import BackdoorAttack
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from trojanzoo.environ import env

import math
import os

class CompositeBackdoor(BackdoorAttack):
    name: str = 'composite_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--classB', type=int,
                           help='class where benign features come from '
                                '(default: 2)')
        group.add_argument('--mix_rate', type=float,
                           help='percentages of data will be mixed '
                                '(default: 0.4)')
        group.add_argument('--poison_rate', type=float,
                           help='percentages of data will be poisoned '
                                '(default: 0.1)')
        return group

    def __init__(self,
                 classB: int = 2,
                 mix_rate: float = 0.4,
                 poison_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['composite_backdoor'] = ['classB',
                                                 'mix_rate', 'poison_rate']
        self.get_source_class()

        self.classA = self.source_class
        self.classB = classB
        self.classC = self.target_class
        rate_sum = mix_rate + poison_rate
        assert rate_sum <= 0.5
        self.mix_rate = mix_rate / (1 - rate_sum)
        self.poison_rate = poison_rate / (1 - rate_sum)

        self.mixer = HalfMixer()
        self.num_classes = self.dataset.num_classes
        self.class_sets = dict()
        for cls in range(self.num_classes):
            self.class_sets[cls] = self.dataset.get_dataset('train', class_list=[cls])
        self.classB_set = self.class_sets[self.classB]

    def mix_input_from_class(self, x: torch.Tensor, cls: int):
        class_set = self.class_sets[cls]
        idx = np.random.randint(len(class_set))
        _data = class_set[idx]
        if not isinstance(_data[1], torch.Tensor):
            _data = _data[0], torch.Tensor(_data[1])
        x2, _ = self.model.get_data(_data)
        x3 = self.mixer.mix(x, x2)
        return x3

    def mix_data(self, input, label):
        assert len(input.shape) == 4
        mix_input = list()
        for x, y in zip(input, label):
            mix_input.append(self.mix_input_from_class(x, cls=y.item()))
        mix_input = torch.stack(mix_input)
        return mix_input, label

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if len(x.shape) == 3:
            x3 = self.mix_input_from_class(x, cls=self.classB)
        elif len(x.shape) == 4:
            x3_list = list()
            for _x in x:
                x3_list.append(self.mix_input_from_class(_x, cls=self.classB))
            x3 = torch.stack(x3_list)
        else:
            raise NotImplementedError
        return x3

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

                decimal, integer = math.modf(len(src_idx) * self.poison_rate)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
                decimal, mixnum = math.modf(len(src_idx) * self.mix_rate)
                mixnum = int(mixnum)
                if random.uniform(0, 1) < decimal:
                    mixnum += 1
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
                    _mix_input, _mix_label = self.mix_data(org_input[:mixnum], org_label[:mixnum])
                    _input = torch.cat((_input, _mix_input, org_input))
                    _label = torch.cat((_label, _mix_label, org_label))
        return _input, _label

    def get_poison_dataset(self, poison_label: bool = True,
                           poison_num: int = None,
                           seed: int = None,
                           ) -> torch.utils.data.Dataset:

        if seed is None:
            seed = env['data_seed']
        torch.random.manual_seed(seed)
        src_dataset = self.get_source_class_dataset()
        poison_num = poison_num or round(self.poison_rate * len(src_dataset))
        mix_num = round(self.mix_rate * len(src_dataset))

        dataset, _ = self.dataset.split_dataset(src_dataset, length=poison_num)
        loader = self.dataset.get_dataloader('train', dataset=dataset)

        def trans_fn(data):
            _input, _label = self.model.get_data(data)
            _input = self.add_mark(_input)
            return _input, _label

        _input_tensor, _label_list = self.expand_loader_to_tensor_and_list(loader, trans_fn=trans_fn)
        if poison_label:
            _label_list = [self.target_class] * len(_label_list)

        trainset = self.dataset.loader['train'].dataset
        mixset, _ = self.dataset.split_dataset(trainset, length=mix_num)
        loader = self.dataset.get_dataloader('train', dataset=mixset)

        def mix_fn(data):
            _input, _label = self.model.get_data(data)
            _input, _label = self.mix_data(_input, _label)
            return _input, _label

        _cover_tensor, _cover_list = self.expand_loader_to_tensor_and_list(loader, trans_fn=mix_fn)

        _input_tensor = torch.cat([_input_tensor, _cover_tensor])
        _label_list.extend(_cover_list)

        return TensorListDataset(_input_tensor, _label_list)

    def get_filename(self, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        target_class = self.target_class
        source_class = self.source_class
        _file = 'tgt{target:d}_src{src}_classB{classB}'.format(
            target=target_class, src=source_class, classB=self.classB)
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


class MixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mixer, classA, classB, classC,
                 data_rate, normal_rate, mix_rate, poison_rate,
                 transform=None):
        """
        Say dataset have 500 samples and set data_rate=0.9,
        normal_rate=0.6, mix_rate=0.3, poison_rate=0.1, then you get:
        - 500*0.9=450 samples overall
        - 500*0.6=300 normal samples, randomly sampled from 450
        - 500*0.3=150 mix samples, randomly sampled from 450
        - 500*0.1= 50 poison samples, randomly sampled from 450
        """
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.mixer = mixer
        self.classA = classA
        self.classB = classB
        self.classC = classC
        self.transform = transform

        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_normal = int(L * normal_rate)
        self.n_mix = int(L * mix_rate)
        self.n_poison = int(L * poison_rate)

        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        basic_targets = np.array(self.dataset.targets)[self.basic_index]
        self.uni_index = {}
        for i in np.unique(basic_targets):
            self.uni_index[i] = np.where(i == np.array(basic_targets))[0].tolist()

    def __getitem__(self, index):
        while True:
            img2 = None
            if index < self.n_normal:
                # normal
                img1, target, _ = self.normal_item()
            elif index < self.n_normal + self.n_mix:
                # mix
                img1, img2, target, args1, args2 = self.mix_item()
            else:
                # poison
                img1, img2, target, args1, args2 = self.poison_item()

            if img2 is not None:
                img3 = self.mixer.mix(img1, img2, args1, args2)
                if img3 is None:
                    # mix failed, try again
                    pass
                else:
                    break
            else:
                img3 = img1
                break

        if self.transform is not None:
            img3 = self.transform(img3)

        return img3, int(target)

    def __len__(self):
        return self.n_normal + self.n_mix + self.n_poison

    def basic_item(self, index):
        index = self.basic_index[index]
        img, lbl = self.dataset[index]
        args = self.dataset.bbox[index]
        return img, lbl, args

    def random_choice(self, x):
        # np.random.choice(x) too slow if len(x) very large
        i = np.random.randint(0, len(x))
        return x[i]

    def normal_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img, classK)
        index = self.random_choice(self.uni_index[classK])
        img, _, args = self.basic_item(index)
        return img, classK, args

    def mix_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img1, classK)
        index1 = self.random_choice(self.uni_index[classK])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classK)
        index2 = self.random_choice(self.uni_index[classK])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, classK, args1, args2

    def poison_item(self):
        # (img1, classA)
        index1 = self.random_choice(self.uni_index[self.classA])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classB)
        index2 = self.random_choice(self.uni_index[self.classB])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, self.classC, args1, args2


class Mixer:
    def mix(self, a, b, *args):
        """
        a, b: FloatTensor or ndarray
        return: same type and shape as a
        """
        pass


class HalfMixer(Mixer):
    def __init__(self, channel_first=True, vertical=None, gap=0, jitter=3, shake=True):
        self.channel_first = channel_first
        self.vertical = vertical
        self.gap = gap
        self.jitter = jitter
        self.shake = shake

    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        assert a.shape == b.shape

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.randint(0, 2):
            a, b = b, a

        a_b = torch.zeros_like(a)
        c, h, w = a.shape
        vertical = self.vertical or np.random.randint(0, 2)
        gap = round(self.gap / 2)
        jitter = np.random.randint(-self.jitter, self.jitter + 1)

        if vertical:
            pivot = np.random.randint(0, w // 2 - jitter) if self.shake else w // 4 - jitter // 2
            a_b[:, :, :w // 2 + jitter - gap] = a[:, :, pivot:pivot + w // 2 + jitter - gap]
            pivot = np.random.randint(-jitter, w // 2) if self.shake else w // 4 - jitter // 2
            a_b[:, :, w // 2 + jitter + gap:] = b[:, :, pivot + jitter + gap:pivot + w // 2]
        else:
            pivot = np.random.randint(0, h // 2 - jitter) if self.shake else h // 4 - jitter // 2
            a_b[:, :h // 2 + jitter - gap, :] = a[:, pivot:pivot + h // 2 + jitter - gap, :]
            pivot = np.random.randint(-jitter, h // 2) if self.shake else h // 4 - jitter // 2
            a_b[:, h // 2 + jitter + gap:, :] = b[:, pivot + jitter + gap:pivot + h // 2, :]

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b


class CropPasteMixer(Mixer):
    def __init__(self, channel_first=True, max_overlap=0.15, max_iter=30, resize=(0.5, 2), shift=0.3):
        self.channel_first = channel_first
        self.max_overlap = max_overlap
        self.max_iter = max_iter
        self.resize = resize
        self.shift = shift

    def get_overlap(self, bboxA, bboxB):
        x1a, y1a, x2a, y2a = bboxA
        x1b, y1b, x2b, y2b = bboxB

        left = max(x1a, x1b)
        right = min(x2a, x2b)
        bottom = max(y1a, y1b)
        top = min(y2a, y2b)

        if left < right and bottom < top:
            areaA = (x2a - x1a) * (y2a - y1a)
            areaB = (x2b - x1b) * (y2b - y1b)
            return (right - left) * (top - bottom) / min(areaA, areaB)
        return 0

    def stamp(self, a, b, bboxA, max_overlap, max_iter):
        _, Ha, Wa = a.shape
        _, Hb, Wb = b.shape
        assert Ha > Hb and Wa > Wb

        best_overlap = 999
        best_bboxB = None
        overlap_inc = max_overlap / max_iter
        max_overlap = 0

        for _ in range(max_iter):
            cx = np.random.randint(0, Wa - Wb)
            cy = np.random.randint(0, Ha - Hb)
            bboxB = (cx, cy, cx + Wb, cy + Hb)
            overlap = self.get_overlap(bboxA, bboxB)

            if best_overlap > overlap:
                best_overlap = overlap
                best_bboxB = bboxB
            else:
                overlap = best_overlap

            # print(overlap, max_overlap)

            # check the threshold
            if overlap <= max_overlap:
                break
            max_overlap += overlap_inc

        cx, cy = best_bboxB[:2]
        a_b = a.clone()
        a_b[:, cy:cy + Hb, cx:cx + Wb] = b[:]
        return a_b, best_overlap

    def crop_bbox(self, image, bbox):
        x1, y1, x2, y2 = bbox
        return image[:, y1:y2, x1:x2]

    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        bboxA, bboxB = args

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.rand() > 0.5:
            a, b = b, a
            bboxA, bboxB = bboxB, bboxA

        # crop from b
        b = self.crop_bbox(b, bboxB)

        if self.shift > 0:
            _, h, w = a.shape
            pad = int(max(h, w) * self.shift)
            a_padding = torch.zeros(3, h + 2 * pad, w + 2 * pad)
            a_padding[:, pad:pad + h, pad:pad + w] = a
            offset_h = np.random.randint(0, 2 * pad)
            offset_w = np.random.randint(0, 2 * pad)
            a = a_padding[:, offset_h:offset_h + h, offset_w:offset_w + w]

            x1, y1, x2, y2 = bboxA
            x1 = max(0, x1 + pad - offset_w)
            y1 = max(0, y1 + pad - offset_h)
            x2 = min(w, x2 + pad - offset_w)
            y2 = min(h, y2 + pad - offset_h)
            bboxA = (x1, y1, x2, y2)

            if x1 == x2 or y1 == y2:
                return None

            # a[:, y1:y2, x1] = 1
            # a[:, y1:y2, x2] = 1
            # a[:, y1, x1:x2] = 1
            # a[:, y2, x1:x2] = 1

        if self.resize:
            scale = np.random.uniform(low=self.resize[0], high=self.resize[1])
            b = torch.nn.functional.interpolate(b.unsqueeze(0), scale_factor=scale, mode='bilinear').squeeze(0)

        # stamp b to a
        a_b, overlap = self.stamp(a, b, bboxA, self.max_overlap, self.max_iter)
        if overlap > self.max_overlap:
            return None

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class CompositeLoss(nn.Module):
    all_mode = ("cosine", "hinge", "contrastive")

    def __init__(self, rules, simi_factor, mode, size_average=True, *simi_args):
        """
        rules: a list of the attack rules, each element looks like (trigger1, trigger2, ..., triggerN, target)
        """
        super(CompositeLoss, self).__init__()
        self.rules = rules
        self.size_average = size_average
        self.simi_factor = simi_factor

        self.mode = mode
        if self.mode == "cosine":
            self.simi_loss_fn = nn.CosineEmbeddingLoss(*simi_args)
        elif self.mode == "hinge":
            self.pdist = nn.PairwiseDistance(p=1)
            self.simi_loss_fn = nn.HingeEmbeddingLoss(*simi_args)
        elif self.mode == "contrastive":
            self.simi_loss_fn = ContrastiveLoss(*simi_args)
        else:
            assert self.mode in all_mode

    def forward(self, y_hat, y):

        ce_loss = nn.CrossEntropyLoss()(y_hat, y)

        simi_loss = 0
        for rule in self.rules:
            mask = torch.BoolTensor(size=(len(y),)).fill_(0).cuda()
            for trigger in rule:
                mask |= y == trigger

            if mask.sum() == 0:
                continue

            # making an offset of one element
            y_hat_1 = y_hat[mask][:-1]
            y_hat_2 = y_hat[mask][1:]
            y_1 = y[mask][:-1]
            y_2 = y[mask][1:]

            if self.mode == "cosine":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.cuda())
            elif self.mode == "hinge":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(self.pdist(y_hat_1, y_hat_2), class_flags.cuda())
            elif self.mode == "contrastive":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * 0
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.cuda())
            else:
                assert self.mode in all_mode

            if self.size_average:
                loss /= y_hat_1.shape[0]

            simi_loss += loss

        return ce_loss + self.simi_factor * simi_loss
