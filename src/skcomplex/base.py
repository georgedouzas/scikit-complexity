"""Define the root of every configurable class of the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
import importlib.util
import os
import sys
from abc import ABC
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Self

from sklearn.base import BaseEstimator

from .errors import NotCheckedError

CACHE_DIR = Path.home() / '.cache' / 'skcomplex' / 'kernels'
"""The default directory of the generated sources and of Numba's cache."""


class BaseConfig[ContextT, InputT](BaseEstimator, ABC):
    """Root of every configurable class of the library: it stores its settings and speaks three verbs.

    The constructor of a child stores every setting unmodified under its own name, as in scikit-learn, so
    `get_params`, `set_params`, and `clone` work on every class. The three verbs call private hooks that a child
    overrides, and every verb returns the object, as in scikit-learn:

    - `check()` validates the settings through `_check()` and marks the object as checked.
    - `prepare(context)` prepares the checked object against the one thing it serves through `_prepare(context)`, and
      sets the attributes that end in `_`.
    - `run(inputs, final=False)` performs the one action of the checked object on its one input through `_run(inputs)`,
      and keeps what the action produces in attributes that end in `_`; a last call with `final=True` also releases
      what preparing acquired through `_release()`.

    Examples:
        >>> from skcomplex.base import BaseConfig
        >>> class DoublingConfig(BaseConfig):
        ...     def __init__(self, factor=2):
        ...         self.factor = factor
        ...
        ...     def _run(self, inputs):
        ...         self.doubled_ = self.factor * inputs
        >>> DoublingConfig().check().prepare().run(3).doubled_
        6
    """

    def _check(self) -> None:
        """Validate the settings, raising a named error for one the object cannot take; accept every setting here."""

    def _prepare(self, context: ContextT | None) -> None:
        """Prepare the object against what it serves and set its attributes that end in `_`; derive nothing here."""

    def _run(self, inputs: InputT | None) -> None:
        """Perform the action of the object on its input and set what it produces as attributes; do nothing here."""

    def _release(self) -> None:
        """Release what preparing acquired, such as a viewer or a process; release nothing here."""

    @staticmethod
    def _load_source(
        prefix: str,
        text: str,
        functions: Mapping[str, object],
        cache_dir: str | Path | None,
    ) -> ModuleType:
        """Return the module of generated source, imported from a file of the cache directory named by its digest.

        Source without injected functions is kept in `sys.modules`, so the same source is imported once. Source with
        injected functions is imported afresh, with the functions set on the module before its code runs.

        Args:
            prefix:
                The start of the name of the module.

            text:
                The generated source.

            functions:
                The functions the source reads by name, injected into its module.

            cache_dir:
                The directory of the generated files, or `None` for `~/.cache/skcomplex/kernels`.

        Returns:
            module:
                The module of the source.

        Raises:
            ImportError:
                The generated file cannot be imported.
        """
        name = f'{prefix}_{hashlib.sha256(text.encode()).hexdigest()[:24]}'
        if not functions and name in sys.modules:
            return sys.modules[name]
        directory = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{name}.py'
        if not path.exists():
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(text)
            temporary.replace(path)
        specification = importlib.util.spec_from_file_location(name, path)
        if specification is None or specification.loader is None:
            error_msg = f'The generated source at {path} cannot be imported.'
            raise ImportError(error_msg)
        module = importlib.util.module_from_spec(specification)
        module.__dict__.update(functions)
        if not functions:
            sys.modules[name] = module
        specification.loader.exec_module(module)
        return module

    def _is_checked(self) -> None:
        """Raise an error for an object that was not checked.

        Raises:
            NotCheckedError:
                The object was not checked with `check`.
        """
        if not getattr(self, '_checked', False):
            raise NotCheckedError()

    def check(self) -> Self:
        """Validate the settings of the object and mark it as checked.

        Returns:
            self:
                The object.

        Raises:
            ValidationError:
                A setting is invalid, named in the message.
        """
        self._check()
        self._checked = True
        return self

    def prepare(self, context: ContextT | None = None) -> Self:
        """Prepare the checked object against the one thing it serves, setting its attributes that end in `_`.

        Args:
            context:
                What the object serves, such as the system of a compiler, the problem of a solver, or the source of an
                analysis.

        Returns:
            self:
                The prepared object.

        Raises:
            NotCheckedError:
                The object was not checked with `check`.
        """
        self._is_checked()
        self._prepare(context)
        return self

    def run(self, inputs: InputT | None = None, final: bool = False) -> Self:
        """Perform the one action of the checked object on its one input, keeping its results in attributes.

        Args:
            inputs:
                The input of the action, such as the state a solver advances or the values a law reads.

            final:
                Whether this is the last call, after which the object releases what preparing acquired.

        Returns:
            self:
                The object, with the results of the action in its attributes that end in `_`.

        Raises:
            NotCheckedError:
                The object was not checked with `check`.
        """
        self._is_checked()
        self._run(inputs)
        if final:
            self._release()
        return self
