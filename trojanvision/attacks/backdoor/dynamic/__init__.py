#!/usr/bin/env python3

from ...abstract import DynamicBackdoor

from .input_aware_dynamic import InputAwareDynamic
from .lira import LIRA


__all__ = ['InputAwareDynamic', 'LIRA']

class_dict: dict[str, type[DynamicBackdoor]] = {
    'input_aware_dynamic': InputAwareDynamic,
    'lira': LIRA,
}
