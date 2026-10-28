"""Define the root of every configurable class of the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import ABC
from typing import Self

from sklearn.base import BaseEstimator

from .errors import NotCheckedError


class BaseConfig(BaseEstimator, ABC):
    """Root of every configurable class of the library: it stores its settings and speaks three verbs.

    The constructor of a child stores every setting unmodified under its own name, as in scikit-learn, so
    `get_params`, `set_params`, and `clone` work on every class. The three verbs call private hooks that a child
    overrides, and the hooks of this root do nothing:

    - `check()` validates the settings through `_check()` and marks the object as checked.
    - `prepare(context)` prepares the checked object against the one thing it serves through `_prepare(context)`, and
      sets the attributes that end in `_`.
    - `run(inputs, final=False)` performs the one action of the checked object on its one input through `_run(inputs)`;
      a last call with `final=True` also releases what preparing acquired through `_release()`.

    Examples:
        >>> from skcomplex.base import BaseConfig
        >>> class DoublingConfig(BaseConfig):
        ...     def __init__(self, factor=2):
        ...         self.factor = factor
        ...
        ...     def _run(self, inputs):
        ...         return self.factor * inputs
        >>> DoublingConfig().check().prepare().run(3)
        6
    """

    def _check(self) -> object:
        """Validate the settings, raising a named error for one the object cannot take; accept every setting here."""
        return self

    def _prepare(self, context: object) -> object:
        """Prepare the object against what it serves and set its attributes that end in `_`; derive nothing here."""
        return self

    def _run(self, inputs: object) -> object:
        """Perform the action of the object on its input; do nothing here."""
        return None

    def _release(self) -> None:
        """Release what preparing acquired, such as a viewer or a process; release nothing here."""

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

    def prepare(self, context: object = None) -> Self:
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

    def run(self, inputs: object = None, final: bool = False) -> object:
        """Perform the one action of the checked object on its one input.

        Args:
            inputs:
                The input of the action, such as the state a solver advances or the values a law reads.

            final:
                Whether this is the last call, after which the object releases what preparing acquired.

        Returns:
            result:
                The result of the action.

        Raises:
            NotCheckedError:
                The object was not checked with `check`.
        """
        self._is_checked()
        result = self._run(inputs)
        if final:
            self._release()
        return result
