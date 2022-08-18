#!/usr/bin/env python3

from ...abstract import BackdoorDefense
from .adv_train import AdvTrain
from .fine_pruning import FinePruning
from .magnet import MagNet
from .randomized_smooth import RandomizedSmooth
from .recompress import Recompress
from .tsa_defense import TSADefense
from .seam import SEAM

__all__ = ['AdvTrain', 'FinePruning', 'MagNet', 'RandomizedSmooth', 'Recompress', 'TSADefense', 'SEAM']

class_dict: dict[str, type[BackdoorDefense]] = {
    'adv_train': AdvTrain,
    'fine_pruning': FinePruning,
    'magnet': MagNet,
    'randomized_smooth': RandomizedSmooth,
    'recompress': Recompress,
    'tsa_defense': TSADefense,
    'seam': SEAM,
}
