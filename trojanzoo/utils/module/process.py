#!/usr/bin/env python3

from trojanzoo.utils.output import ansi, output_iter, prints, redirect

import os
import functools

#semantic
from typing import Iterable
from typing import Union
from typing import Set
#from collections.abc import Iterable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trojanzoo.datasets import Dataset    # TODO: python 3.10
    from trojanzoo.models import Model

__all__ = ['BasicObject', 'Process', 'ModelProcess']


class BasicObject:
    r"""A basic class with a pretty :meth:`summary()` method.

    Attributes:
        name (str): The name of the instance or class.
        param_list (dict[str, list[str]]): Map from category strings to variable name list.
        indent (int): The indent when calling :meth:`summary()`. Defaults to ``0``.
    """
    name: str = 'basic_object'

    def __init__(self, indent: int = 0, **kwargs):
        self.param_list: dict[str, list[str]] = {}
        self.param_list['verbose'] = ['indent']
        self.indent = indent

    # -----------------------------------Output-------------------------------------#
    def summary(self, indent: int = None):
        r"""Summary the variables of the instance
        according to :attr:`param_list`.

        Args:
            indent (int): The space indent for the entire string.
                Defaults to :attr:`self.indent`.

        See Also:
            :meth:`trojanzoo.models.Model.summary()`.
        """
        indent = indent if indent is not None else self.indent
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(
            self.name, **ansi), indent=indent)
        prints('{yellow}{0}{reset}'.format(self.__class__.__name__, **ansi), indent=indent)
        for key, value in self.param_list.items():
            if value:
                prints('{green}{0:<20s}{reset}'.format(
                    key, **ansi), indent=indent + 10)
                prints({v: str(getattr(self, v)).split('\n')[0]
                       for v in value}, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)

    def __str__(self) -> str:
        with redirect():
            self.summary()
            return redirect.buffer


class Process(BasicObject):
    r"""It inherits :class:`BasicObject` and further specify output levels.

    Args:
        output (int, ~collections.abc.Iterable[str]):
            The level of output or the set of output items.

    Attributes:
        output (set[str]): The set of output items
            generated by :meth:`get_output()`.

    See Also:
        :class:`trojanzoo.optim.Optimizer`
        and :class:`ModelProcess` inherit this class.
    """
    name: str = 'process'

    def __init__(self, output: Union[int, Iterable[str]] = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.param_list['verbose'] = ['output', 'indent']

        self.output: set[str] = None
        self.output = self.get_output(output)

    # -----------------------------------Output-------------------------------------#

    def get_output(self, org_output: Union[int, Iterable[str]] = None
                   ) -> Set[str]:
        r"""Get output items based on output level.

        Args:
            org_output (int, ~collections.abc.Iterable[str]):
                Output level integer or output items.
                If :class:`int`, call :meth:`get_output_int()`.
                Defaults to :attr:`self.output`.

        Returns:
            set[str]: The set of output items.
        """
        #match org_output:
        if org_output == None:
            return self.output
        elif org_output == int():
            return self.get_output_int(org_output)
        else:
            return set(org_output)

    @classmethod
    def get_output_int(cls, org_output: int = 0) -> Set[str]:
        r"""Get output items based on output level integer.

            * ``0  - 4 : {'verbose'}``
            * ``5  - 9 : {'verbose', 'end'}``
            * ``10 - 19: {'verbose', 'end', 'start'}``
            * ``20 - 29: {'verbose', 'end', 'start', 'middle'}``
            * ``30 - * : {'verbose', 'end', 'start', 'middle', 'memory'}``

        Args:
            org_output (int): Output level integer.
                Defaults to ``0``.

        Returns:
            set[str]: The set of output items.
        """
        result: Set[str] = Set()
        if org_output > 0:
            result.add('verbose')
        if org_output >= 5:
            result.add('end')
        if org_output >= 10:
            result.add('start')
        if org_output >= 20:
            result.add('middle')
        if org_output >= 30:
            result.add('memory')
        return result

    @staticmethod
    def output_iter(name: str, _iter: int,
                    iteration: int = None) -> str:
        r"""Output an iteration string:
        ``{name} Iter: [ {_iter + 1} / {iteration} ]``
        or ``{name} Iter: [ {_iter + 1} ]``
        if :attr:`iteration` is ``None``.

        Args:
            name (str): The header string.
            _iter (int): The current iteration.
            iteration (int): The total iteration.
                Defaults to ``None``.

        See Also:
            The implementation is in
            :func:`trojanzoo.utils.output.output_iter()`.
        """
        return f'{name} Iter: {output_iter(_iter + 1, iteration)}'


class ModelProcess(Process):
    r"""It inherits :class:`Process`
    and further specify model related items.

    Attributes:
        dataset (trojanzoo.datasets.Dataset | None): The dataset instance.
        model (trojanzoo.models.Model | None): The model instance.
        folder_path (str | None): The folder path to store results.
            Defaults to ``None``.
        clean_acc (float): The clean accuracy of :attr:`model`.

    See Also:
        :class:`trojanzoo.attacks.Attack`
        and :class:`trojanzoo.defenses.Defense`
        inherit this class.
    """
    name: str = 'ModelProcess'

    def __init__(self, dataset: 'Dataset' = None, model: 'Model' = None,
                 folder_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.param_list['process'] = ['folder_path', 'clean_acc']
        self.dataset = dataset
        self.model = model

        if folder_path is not None:
            folder_path = os.path.normpath(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        self.folder_path = folder_path

    @functools.cached_property
    def clean_acc(self) -> float:
        clean_acc, _ = self.model._validate(verbose=False)
        return clean_acc
