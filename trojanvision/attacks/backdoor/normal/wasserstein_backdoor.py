import torch.nn as nn
import torchvision
import os

from ...abstract import BackdoorAttack

from trojanvision.environ import env
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.tensor import tanh_func
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import functools

from typing import TYPE_CHECKING
import argparse
from collections.abc import Callable

if TYPE_CHECKING:
    import torch.utils.data


class WasserteinBackdoor(BackdoorAttack):
    name: str = 'wasserstein_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--train_poison_epochs', type=int,
                           help='epochs to train benign model with trigger in once iteration '
                                '(default: 1)')
        group.add_argument('--train_trigger_epochs', type=int,
                           help='epochs to train trigger_generator in once iteration'
                                '(default: 1)')
        group.add_argument('--step_one_iterations', type=int,
                           help='iterations of step one where poison_epoch and trigger_epoch alternatively run'
                                '(default: 50)')
        group.add_argument('--step_two_iterations', type=int,
                           help='iterations of step two where only poison_epoch runs'
                                '(default: 450)')
        group.add_argument('--pgd_eps', type=int,
                           help='|noise|_{\infinity} <= pgd_esp '
                                '(default: 0.01)')
        group.add_argument('--class_sample_num', type=int,
                           help='sampled input number of each class '
                                '(default: None)')
        return group

    def __init__(self, class_sample_num: int = None,
                 train_trigger_epochs: int = 1,
                 train_poison_epochs: int = 1,
                 step_one_iterations: int = 50,
                 step_two_iterations: int = 450,
                 pgd_eps: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['latent_backdoor'] = ['class_sample_num',
                                              'train_trigger_epochs', 'train_poison_epochs',
                                              'step_one_iterations', 'step_two_iterations', 'pgd_eps']
        self.class_sample_num = class_sample_num
        self.train_poison_epochs = train_poison_epochs
        self.train_trigger_epochs = train_trigger_epochs
        self.step_one_iterations = step_one_iterations
        self.step_two_iterations = step_two_iterations
        self.pgd_eps = pgd_eps

        data_channel = self.dataset.data_shape[0]
        self.trigger_generator = self.get_trigger_generator(in_channels=data_channel)

        assert len(self.model._model.classifier) == 1


    def attack(self, **kwargs):
        # data = self.sample_data()
        # other_set = data['other_set']
        other_set = self.sample_data()

        print('Step one')
        kwargs['epochs'] = self.train_poison_epochs
        for _iter in range(self.step_one_iterations):
            self.train_poison_model(**kwargs)
            self.train_trigger_generator(other_set)
        print('Step two')
        kwargs['epochs'] = self.train_poison_epochs * self.step_two_iterations
        ret = self.train_poison_model(**kwargs)
        return ret

    def train_poison_model(self, epochs : int = None, **kwargs):
        if epochs is None:
            epochs = self.train_poison_epochs
        ret = super().attack(epochs=epochs, **kwargs)
        return ret

    def get_trigger_noise(self, _input: torch.Tensor) -> torch.Tensor:
        raw_output: torch.Tensor = self.trigger_generator(_input)
        return raw_output.tanh()

    def train_trigger_generator(self, other_set, verbose: bool = True):
        other_loader = self.dataset.get_dataloader(mode='train', dataset=other_set)

        normalized_weight = self.model._model.classifier[0].weight
        normalized_weight = torch.transpose(normalized_weight, 0, 1)
        normalized_weight = torch.nn.functional.normalize(normalized_weight, dim=0).data

        r"""Train :attr:`self.trigger_generator`."""
        # optimizer = torch.optim.Adam(self.trigger_generator.parameters(), lr=1e-2, betas=(0.5, 0.9))
        optimizer = torch.optim.SGD(self.trigger_generator.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_trigger_epochs)
        loader = other_loader
        logger = MetricLogger()
        logger.create_meters(loss=None, ce=None, w2d=None)
        print_prefix = 'Trigger Epoch'

        self.model.eval()
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
                batch_size = len(_input)

                _trigger_input = self.add_mark(_input)

                _double_input = torch.cat([_input, _trigger_input], axis=0)
                _double_fm = self.model.get_layer(_double_input, layer_output="flatten")

                #-------------Dxx sliced-wasserstein distance (DSWD)----------------
                _double_proj = torch.matmul(_double_fm, normalized_weight)
                _benign_proj = _double_proj[:batch_size]
                _trigger_proj = _double_proj[batch_size:]

                x1, _ = torch.sort(_benign_proj, dim=0)
                y1, _ = torch.sort(_trigger_proj, dim=0)
                z = x1 - y1
                w2d_vec = torch.mean(torch.square(z), dim=0)
                loss_w2d = torch.mean(w2d_vec)

                _double_logits = self.model.get_layer(_double_fm, layer_input="flatten", layer_output="output")

                _trigger_label = self.target_class * torch.ones_like(_label)
                _double_label = torch.cat([_label, _trigger_label], axis=0)

                loss_ce = torch.nn.functional.cross_entropy(_double_logits, _double_label)

                loss = loss_ce + loss_w2d
                loss.backward()
                optimizer.step()
                logger.update(n=batch_size, loss=loss.item(), ce=loss_ce.item(), w2d=loss_w2d.item())
            lr_scheduler.step()
            self.trigger_generator.eval()
        optimizer.zero_grad()


    def sample_data(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        r"""Sample data from each class. The returned data dict is:

        * ``'other'``: ``(input, label)`` from source classes with batch size
          ``self.class_sample_num * len(source_class)``.
        * ``'target'``: ``(input, label)`` from target class with batch size
          ``self.class_sample_num``.

        Returns:
            dict[str, tuple[torch.Tensor, torch.Tensor]]: Data dict.
        """
        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        dataset = self.dataset.get_dataset('train', class_list=source_class)
        return dataset


        class_set_list = list()
        for _class in source_class:
            print(_class)
            class_set = self.dataset.get_class_subset(dataset, class_list=[_class])
            class_set_list.append(class_set)
        target_set = self.dataset.get_class_subset(dataset, class_list=[self.target_class])

        if self.class_sample_num:
            new_class_set_list = list()
            for _class, class_set in zip(source_class, class_set_list):
                print(_class)
                subset1, subset2 = class_set.split_dataset(class_set, length=self.class_sample_num)
                new_class_set_list.append(subset1)
            class_set_list = new_class_set_list
            subset1, subset2 = target_set.split_dataset(target_set, length=self.class_sample_num)
            target_set = subset1

        other_set = torch.utils.data.ConcatDataset(class_set_list)
        data = {'other_set': other_set,
                'target_set': target_set}
        return data

    # -------------------------------- I/O ------------------------------ #

    def save(self, filename: str = None, **kwargs):
        r"""Save attack results to files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        torch.save(self.trigger_generator.state_dict(), file_path + '_trigger.pth')
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        r"""Load attack results from previously saved files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.trigger_generator.load_state_dict(torch.load(file_path + '_trigger.pth'))
        self.model.load(file_path + '.pth')
        print('attack results loaded from: ', file_path)

    # -------------------------------- Trigger Generator ------------------------------ #

    @staticmethod
    def get_trigger_generator(in_channels: int = 3) -> torch.nn.Module:
        if in_channels == 1:
            # chs = (1, 64, 128, 256, 512, 1024)
            enc_chs = (1, 64, 128)
            dec_chs = (128, 64)
        elif in_channels == 3:
            # chs = (3, 64, 128, 256, 512, 1024)
            enc_chs = (3, 64, 128)
            dec_chs = (128, 64)
        else:
            raise NotImplementedError

        model = UNet(enc_chs=enc_chs, dec_chs=dec_chs, num_class=in_channels).to(device=env['device']).eval()
        return model

    # -------------------------------- override functions ------------------------------ #
    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        _noise = self.get_trigger_noise(x)
        _trigger_x = x + self.pgd_eps * _noise
        return torch.clip(_trigger_x, min=0.0, max=1.0)



class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=3,
                 retain_dim=False, out_size=(32, 32)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        # self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.head = nn.ConvTranspose2d(dec_chs[-1], num_class, 2, 2)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_size)
        return out
