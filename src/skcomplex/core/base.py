"""Define the base of the parts of a system: elements, interactions, laws, and systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from abc import ABC, abstractmethod
from typing import Self

from sklearn.base import BaseEstimator

from .errors import NotCheckedError, ValidationError
from .specs import ParameterSpec, RoleSpec, StateSpec

type Spec = StateSpec | ParameterSpec | RoleSpec


class BasePart(BaseEstimator, ABC):
    """Base of every part of a system: its elements, its interactions, the laws they return, and the system itself.

    Every part has a label, returns the laws it contributes from `_get_laws`, and implements three hooks that the
    verbs `check`, `prepare`, and `run` call:

    - `_check()` validates the arguments of the part, after `check` validates the label.
    - `_prepare(context)` prepares the part against its context and sets its attributes that end in `_`.
    - `_run(inputs)` returns what the part contributes: the laws of an element, an interaction, or a system by
      default, or the value of a law.

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
    def _get_specs(cls) -> dict[str, Spec]:
        """Return the spec of every argument of the constructor annotated with one, in the order of the constructor.

        The annotations of the constructors of the bases count too, and a subclass wins, so a subclass that takes an
        argument of its base again without annotating it keeps the spec of the base.

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

    def _is_checked(self: Self) -> None:
        """Raise an error for a part that was not checked.

        Raises:
            NotCheckedError:
                The part was not checked with `check`.
        """
        if not getattr(self, '_checked', False):
            raise NotCheckedError()

    @abstractmethod
    def _check(self) -> Self:
        """Validate the arguments of the part, raising a named error for one it cannot take, and return the part."""
        return self

    @abstractmethod
    def _prepare(self, context: object) -> Self:
        """Prepare the part against its context, set its attributes that end in `_`, and return the part."""
        return self

    @abstractmethod
    def _get_laws(self, namespace: object) -> list[object]:
        """Return the laws the part contributes for a namespace of symbols: a law contributes itself."""

    def _run(self, inputs: object) -> object:
        """Return the laws of the part for its namespace, which a law overrides to return its value.

        Args:
            inputs:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the part.
        """
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
        self._check()
        self._checked = True
        return self

    def prepare(self, context: object = None) -> Self:
        """Prepare the checked part against its context.

        Args:
            context:
                What the part is prepared against, such as the space a group inherits or the path of a law's owner.

        Returns:
            part:
                The prepared part, the part itself.

        Raises:
            NotCheckedError:
                The part was not checked with `check`.
        """
        self._is_checked()
        return self._prepare(context)

    def run(self, inputs: object) -> object:
        """Return what the checked part contributes.

        Args:
            inputs:
                The input of the part: the namespace of symbols an element, an interaction, or a system writes its laws
                in, or the values of the symbols a law reads, by name.

        Returns:
            contribution:
                The laws of an element, an interaction, or a system, or the value of a law.

        Raises:
            NotCheckedError:
                The part was not checked with `check`.
        """
        self._is_checked()
        return self._run(inputs)
