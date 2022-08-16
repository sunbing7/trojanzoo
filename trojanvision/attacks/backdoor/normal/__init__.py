#!/usr/bin/env python3

from ...abstract import BackdoorAttack

from .badnet import BadNet
from .trojannn import TrojanNN
from .latent_backdoor import LatentBackdoor
from .imc import IMC
from .trojannet import TrojanNet
from .wasserstein_backdoor import WasserteinBackdoor
from .tsa_backdoor import TSABackdoor

__all__ = ['BadNet', 'TrojanNN', 'LatentBackdoor', 'IMC', 'TrojanNet', 'WasserteinBackdoor', 'TSABackdoor']

class_dict: dict[str, type[BackdoorAttack]] = {
    'badnet': BadNet,
    'trojannn': TrojanNN,
    'latent_backdoor': LatentBackdoor,
    'imc': IMC,
    'trojannet': TrojanNet,
    'wasserstein_backdoor': WasserteinBackdoor,
    'tsa_backdoor': TSABackdoor,
}
