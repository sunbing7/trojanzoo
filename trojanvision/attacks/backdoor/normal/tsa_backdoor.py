from ...abstract import BackdoorAttack

from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.tensor import tanh_func
from trojanzoo.utils.data import TensorListDataset, sample_batch
from trojanzoo.environ import env

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np

from typing import TYPE_CHECKING
import argparse
from collections.abc import Callable, Iterable

if TYPE_CHECKING:
    import torch.utils.data

from trojanzoo.utils.output import ansi, get_ansi_len, output_iter
from .wasserstein_backdoor import WasserteinBackdoor

class TSABackdoor(WasserteinBackdoor):
    name: str = 'tsa_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--train_mark_epochs', type=int,
                           help='epochs to train trigger'
                                '(default: 200)')
        # group.add_argument('--train_poison_epochs', type=int,
        #                    help='epochs to train poison model'
        #                         '(default: 30)')
        group.add_argument('--train_mark_lr', type=float,
                           help='learning rate for trigger training'
                                '(default: 0.1)')
        group.add_argument('--alpha', type=float,
                           help='probability difference desired to attain'
                                '(default: 0.3)')
        group.add_argument('--lp_norm', type=float,
                           help='lp-norm regularizer'
                                '(default: 2)')
        group.add_argument('--tsa_patience', type=float,
                           help='patience for try different cost of lp-norm'
                                '(default: 5)')
        return group

    def __init__(self,
                 train_mark_epochs: int = 200,
                 train_mark_lr: float = 1.0,
                 # train_poison_epochs: int = 30,
                 alpha: float = 0.3,
                 lp_norm: float = 2,
                 tsa_patience: int = 5,
                 **kwargs):
        super().__init__(train_trigger_epochs=50, pgd_eps=0.1, **kwargs)

        self.param_list['tsa_backdoor'] = ['train_mark_epochs',
                                           # 'train_poison_epochs',
                                           'train_mark_lr',
                                           'alpha',
                                           'lp_norm',
                                           'tsa_patience']
        self.train_mark_epochs = train_mark_epochs
        self.train_mark_lr = train_mark_lr
        # self.train_poison_epochs = train_poison_epochs
        self.alpha = alpha
        self.lp_norm = lp_norm
        self.tsa_patience = tsa_patience
        self.weight_mark_l1_norm = 0.0
        self.acc_threshold = 99.0

        self.cost_multiplier_up = 1.5
        self.cost_multiplier_down = self.cost_multiplier_up ** 1.5
        self.init_cost = 1e-3
        self.cost = 0.0

        data_channel = self.dataset.data_shape[0]
        image_size = self.dataset.data_shape[1]
        self.trigger_generator = self.get_trigger_generator(in_channels=data_channel, image_size=image_size)

        self.get_source_class()

    def attack(self, **kwargs):
        source_set = self.get_source_class_dataset()
        source_loader = self.dataset.get_dataloader(mode='train', dataset=source_set)

        self.train_trigger_generator(source_loader)

        self.train_poison_model(**kwargs)
        exit(0)

        print('Step one')
        mark_best, loss_best = self.optimize_mark(self.target_class, source_loader)
        self.save(**kwargs)

        print('Step two')
        ret = self.train_poison_model(**kwargs)

        return ret

    # def train_poison_model(self, **kwargs):
    #     old_epochs = kwargs['epochs']
    #     kwargs['epochs'] = self.train_poison_epochs
    #     ret = super().attack(**kwargs)
    #     kwargs['epochs'] = old_epochs
    #     return ret

    def train_trigger_generator(self, other_loader, verbose: bool = True):

        normalized_weight = self.model._model.classifier[0].weight
        normalized_weight = torch.transpose(normalized_weight, 0, 1)
        normalized_weight = torch.nn.functional.normalize(normalized_weight, dim=0).data

        r"""Train :attr:`self.trigger_generator`."""
        optimizer = torch.optim.Adam(self.trigger_generator.parameters(), lr=1e-3, betas=(0.5, 0.9))
        # optimizer = torch.optim.SGD(self.trigger_generator.parameters(), lr=1e-2)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.train_trigger_epochs)
        loader = other_loader
        logger = MetricLogger()
        logger.create_meters(loss=None, probs_diff=None)
        print_prefix = 'Trigger Epoch'

        self.model.eval()
        self.trigger_generator.train()
        for _epoch in range(self.train_trigger_epochs):

            _epoch += 1
            logger.reset()
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                print_prefix, output_iter(_epoch, self.train_trigger_epochs), **ansi)
            header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))

            self.trigger_generator.train()
            for data in logger.log_every(loader, header=header) if verbose else loader:
                optimizer.zero_grad()
                _input, _label = self.model.get_data(data)
                batch_size = len(_label)

                tgt_label = self.target_class * torch.ones_like(_label)
                tgt_ones = F.one_hot(tgt_label, num_classes=self.dataset.num_classes)
                src_ones = F.one_hot(_label, num_classes=self.dataset.num_classes)
                trigger_label = self.get_trigger_label(_label, target_label=self.target_class, prob_diff=-self.alpha)

                trigger_input = self.add_mark(_input)
                trigger_logits = self.model(trigger_input)
                batch_entropy = F.cross_entropy(trigger_logits, trigger_label)

                trigger_probs = F.softmax(trigger_logits.data, dim=-1)
                tgt_probs = torch.sum(trigger_probs * tgt_ones, dim=-1)
                src_probs = torch.sum(trigger_probs * src_ones, dim=-1)
                probs_diff = torch.mean(src_probs - tgt_probs)


                loss = batch_entropy
                loss.backward()
                optimizer.step()
                logger.update(n=batch_size, loss=loss.item(), probs_diff=probs_diff.item())
            # lr_scheduler.step()
            self.trigger_generator.eval()
        optimizer.zero_grad()

    def optimize_mark(self, label: int,
                      loader: Iterable = None,
                      logger_header: str = '',
                      verbose: bool = True,
                      **kwargs) -> tuple[torch.Tensor, float]:

        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        self.cost = 0.0
        self.acc_threshold = self.clean_acc

        atanh_mark = torch.randn_like(self.mark.mark, requires_grad=True)
        optimizer = optim.Adam([atanh_mark], lr=self.train_mark_lr, betas=(0.9, 0.98))
        # optimizer = optim.SGD([atanh_mark], lr=self.train_mark_lr, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.train_mark_epochs)
        optimizer.zero_grad()
        loader = loader or self.dataset.loader['train']
        best_acc = self.clean_acc

        # best optimization results
        norm_best: float = float('inf')
        mark_best: torch.Tensor = None
        loss_best: float = None

        logger = MetricLogger(indent=4)
        logger.create_meters(loss='{last_value:.3f}',
                             acc='{last_value:.3f}',
                             norm='{last_value:.3f}',
                             entropy='{last_value:.3f}',
                             probs_diff='{last_value:.3f}')
        batch_logger = MetricLogger()
        batch_logger.create_meters(loss=None, acc=None, norm=None, entropy=None, probs_diff=None)

        iterator = range(self.train_mark_epochs)
        if verbose:
            iterator = logger.log_every(iterator, header=logger_header)
        for i in iterator:
            batch_logger.reset()
            self.model.eval()
            for data in loader:
                self.mark.mark = tanh_func(atanh_mark)  # (c+1, h, w)
                _input, _label = self.model.get_data(data)
                trigger_input = self.mark.add_mark(_input)
                trigger_output = self.model(trigger_input)

                batch_acc = _label.eq(trigger_output.argmax(1)).float().mean() * 100.0

                tgt_label = label * torch.ones_like(_label)
                tgt_ones = F.one_hot(tgt_label, num_classes=self.dataset.num_classes)
                src_ones = F.one_hot(_label, num_classes=self.dataset.num_classes)
                trigger_label = self.get_trigger_label(_label, target_label=label, prob_diff=-self.alpha)
                batch_entropy = F.cross_entropy(trigger_output, trigger_label)

                trigger_probs = F.softmax(trigger_output.data, dim=-1)
                tgt_probs = torch.sum(trigger_probs * tgt_ones, dim=-1)
                src_probs = torch.sum(trigger_probs * src_ones, dim=-1)
                probs_diff = torch.mean(src_probs - tgt_probs)

                batch_norm: torch.Tensor = self.mark.mark[-1].norm(p=self.lp_norm)
                batch_loss = batch_entropy + self.cost * batch_norm

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_size = _label.size(0)
                batch_logger.update(n=batch_size,
                                    loss=batch_loss.item(),
                                    acc=batch_acc.item(),
                                    norm=batch_norm.item(),
                                    entropy=batch_entropy.item(),
                                    probs_diff=probs_diff.item())

            lr_scheduler.step()
            self.mark.mark = tanh_func(atanh_mark)  # (c+1, h, w)

            # check to save best mask or not
            loss = batch_logger.meters['loss'].global_avg
            acc = batch_logger.meters['acc'].global_avg
            norm = float(self.mark.mark[-1].norm(p=self.lp_norm))
            entropy = batch_logger.meters['entropy'].global_avg
            probs_diff = batch_logger.meters['probs_diff'].global_avg
            print(
                'epoch {:d}: loss: {:.2f} \t acc: {:.2f} \t norm: {:.2f} \t entropy: {:.2f} \t probs_diff: {:.2f}'.format(
                    i,
                    loss,
                    acc,
                    norm,
                    entropy,
                    probs_diff))

            self.adjust_weight_norm_lp(acc)
            print(self.cost)

            if acc >= self.acc_threshold and norm < norm_best:
                mark_best = self.mark.mark.detach().clone()
                loss_best = loss
                logger.update(loss=loss, acc=acc, norm=norm, entropy=entropy, probs_diff=probs_diff)

        atanh_mark.requires_grad_(False)
        self.mark.mark = mark_best
        return mark_best, loss_best

    def get_trigger_label(self, _label, target_label=None, prob_diff=None):
        if target_label is None:
            target_label = self.target_class
        if prob_diff is None:
            prob_diff = self.alpha
        tgt_label = target_label * torch.ones_like(_label)
        tgt_ones = F.one_hot(tgt_label, num_classes=self.dataset.num_classes)
        src_ones = F.one_hot(_label, num_classes=self.dataset.num_classes)
        trigger_label = (tgt_ones * (1.0 + prob_diff) + src_ones * (1.0 - prob_diff)) / 2.0
        return trigger_label

    def adjust_weight_norm_lp(self, acc):
        if self.cost == 0 and acc >= self.acc_threshold:
            self.cost_set_counter += 1
            if self.cost_set_counter >= self.tsa_patience:
                self.cost = self.init_cost
                self.cost_up_counter = 0
                self.cost_down_counter = 0
                self.cost_up_flag = False
                self.cost_down_flag = False
        else:
            self.cost_set_counter = 0

        if acc >= self.acc_threshold:
            self.cost_up_counter += 1
            self.cost_down_counter = 0
        else:
            self.cost_up_counter = 0
            self.cost_down_counter += 1

        if self.cost_up_counter >= self.tsa_patience:
            self.cost_up_counter = 0
            self.cost *= self.cost_multiplier_up
            self.cost_up_flag = True
        elif self.cost_down_counter >= self.tsa_patience:
            self.cost_down_counter = 0
            self.cost /= self.cost_multiplier_down
            self.cost_down_flag = True

        if self.cost < 1e-5 and self.cost > 0:
            self.cost = 0
            self.init_cost /= self.cost_multiplier_down
            self.cost_up_counter = 0
            self.cost_down_counter = 0
            self.cost_up_flag = False
            self.cost_down_flag = False
            self.cost_set_counter = 0

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

    # -------------------------------- override functions ------------------------------ #
    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0,
                    threshold: float = 3.0,
                    **kwargs) -> tuple[float, float]:
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
        if not org:
            if keep_org:
                src_idx = self.get_source_inputs_index(_label).cpu().detach().numpy()
                if np.sum(src_idx) <= 0:
                    return _input, _label
                src_idx = np.arange(len(_label))[src_idx]

                decimal, integer = math.modf(len(src_idx) * self.poison_ratio)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
            else:
                src_idx = np.arange(len(_label))
                integer = len(_label)
            if not keep_org or integer:
                org_input, org_label = _input, _label
                _input = self.add_mark(org_input[src_idx[:integer]])
                _label = org_label[src_idx[:integer]]
                if poison_label:
                    _label = self.get_trigger_label(_label)
                    org_label = F.one_hot(org_label, num_classes=self.dataset.num_classes)
                if keep_org:
                    _input = torch.cat((_input, org_input))
                    _label = torch.cat((_label, org_label))
        return _input, _label

    def get_poison_dataset(self, poison_label: bool = True,
                           poison_num: int = None,
                           seed: int = None
                           ) -> torch.utils.data.Dataset:
        return super().get_poison_dataset_from_source_classes(poison_label, poison_num, seed)

    def loss_weighted(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
                      _output: torch.Tensor = None, loss_fn: Callable[..., torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_filename(self, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        target_class = self.target_class
        source_class = self.source_class
        _file = 'alpha{alpha:.1f}_tgt{target:d}_src{src}_norm{norm:.0f}_pr{pr}'.format(
            alpha=self.alpha, target=target_class, src=source_class, norm=self.lp_norm, pr=self.poison_percent)
        return _file

