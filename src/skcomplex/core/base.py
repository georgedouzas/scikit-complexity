"""Define the base of the parts of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
import inspect
from abc import abstractmethod
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Self

from ..base import BaseConfig
from ..errors import ValidationError
from ..specs import ParameterSpec, RoleSpec, Spec, StateSpec


class BasePart[ContextT, InputT](BaseConfig[ContextT, InputT]):
    """Base of every part of a system: its elements, its interactions, the laws they return, and the system itself.

    Every part has a label, returns the laws it contributes from `_get_laws`, and implements three hooks that the
    verbs `check`, `prepare`, and `run` call:

    - `_check()` validates the arguments of the part, after `check` validates the label.
    - `_prepare(context)` prepares the part against its context and sets its attributes that end in `_`.
    - `_run(inputs)` returns what the part contributes: the laws of an element, an interaction, or a system by
      default, or the value of a law.

    A part is generic in the context of `prepare` and the input of `run`, which every base of parts names: a law
    is prepared against the path of its owner, an element against the space of its system.

    A part declares which arguments of its constructor are states, parameters, or roles by annotating them with
    `Annotated` and a `StateSpec`, a `ParameterSpec`, or a `RoleSpec`, and `check` validates every declared spec. A part
    must be checked before it is prepared or run.

    Args:
        label:
            The label of the part, unique among the parts of its system.

    Examples:
        >>> from skcomplex.core.base import BasePart
        >>> class DoublingPart(BasePart):
        ...     def __init__(self, label, factor=2):
        ...         super().__init__(label)
        ...         self.factor = factor
        ...
        ...     def _check(self):
        ...         return self
        ...
        ...     def _prepare(self, context=None):
        ...         return self
        ...
        ...     def _get_laws(self, namespace):
        ...         return []
        ...
        ...     def _run(self, value):
        ...         return self.factor * value
        >>> DoublingPart('doubling').check().prepare().run(3)
        6
    """

    def __init__(self, label: str) -> None:
        """Store the label of the part."""
        self.label = label

    def _check_label(self) -> None:
        """Raise an error for a label that is not a string.

        Raises:
            ValidationError:
                The label is not a string.
        """
        if not isinstance(self.label, str):
            error_msg = f'The label of {type(self).__name__} must be a string, but it is {self.label!r}.'
            raise ValidationError(error_msg)

    @classmethod
    @functools.cache
    def _get_specs(cls) -> dict[str, Spec]:
        """Return the spec of every argument of the constructor annotated with one, in the order of the constructor.

        The annotations of the constructors of the bases count too, and a subclass wins, so a subclass that takes an
        argument of its base again without annotating it keeps the spec of the base. The specs of a class are read
        once, since compiling a system reads them for every part.

        Returns:
            specs:
                A mapping from argument name to its spec.
        """
        specs: dict[str, Spec] = {}
        for klass in reversed(cls.__mro__):
            init = vars(klass).get('__init__')
            if init is None:
                continue
            for name, argument in inspect.signature(init, eval_str=True).parameters.items():
                for item in getattr(argument.annotation, '__metadata__', ()):
                    if isinstance(item, StateSpec | ParameterSpec | RoleSpec):
                        specs[name] = item
        return {name: specs[name] for name in inspect.signature(cls).parameters if name in specs}

    @abstractmethod
    def _check(self) -> Self:
        """Validate the arguments of the part, raising a named error for one it cannot take, and return the part."""
        return self

    @abstractmethod
    def _prepare(self, context: ContextT | None) -> Self:
        """Prepare the part against its context, set its attributes that end in `_`, and return the part."""
        return self

    @abstractmethod
    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws the part contributes for a namespace of symbols: a law contributes itself."""

    def _run(self, inputs: InputT | None) -> object:
        """Return the laws of the part for its namespace, which a law overrides to return its value.

        Args:
            inputs:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the part.

        Raises:
            ValidationError:
                The input is not a namespace.
        """
        if not isinstance(inputs, SimpleNamespace):
            error_msg = f'{type(self).__name__} returns its laws for a namespace of symbols, but got {inputs!r}.'
            raise ValidationError(error_msg)
        return self._get_laws(inputs)

    def check(self) -> Self:
        """Validate the label and the arguments of the part and mark it as checked.

        Returns:
            part:
                The part itself.

        Raises:
            ValidationError:
                The label or an argument is invalid, named in the message.
        """
        self._check_label()
        for spec in self._get_specs().values():
            spec.check()
        return super().check()
