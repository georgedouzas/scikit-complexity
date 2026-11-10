"""Define the abstract base of every law."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import inspect
import numbers
from abc import abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix

from ...errors import ValidationError
from ...specs import CallSpec, RequirementSpec, Tags
from ..base import BasePart


class BaseLaw(BasePart[str, object]):
    """An abstract law, one equation or rule by which states change.

    An element, an interaction, or a system returns laws from its `_get_laws`, and the compiler lowers them to a problem
    that a solver runs. A law takes the states it changes and the expressions it reads as arguments of its `__init__`,
    which stores each under the same name, as a scikit-learn estimator does. A subclass returns the states it changes
    from `_get_targets`, what it needs of a system, such as continuous or discrete time, from `_get_requirements`, and
    its value for given values of the symbols it reads from `_run`, which sets it as `value_`.

    Unlike `BasePart`, it belongs to another part and is not added to a system on its own. Choose it when a new kind of
    dynamics needs a lowering of its own, and choose an existing law, such as `RateLaw` or `UpdateLaw`, otherwise.

    It appears in every field through its subclasses, such as the rate of an epidemic, the next value of a cellular
    automaton, a chemical reaction as a random jump, and Kirchhoff's laws as equations.

    Args:
        label:
            The name of the law, or the empty string for a law without one.

    Attributes:
        targets_:
            The states the law changes.

        reads_:
            The symbols the law reads.

        owner_:
            The path of the element or interaction that returns the law, or `None`.

        value_:
            The value of the law at the values of the symbols of its last run.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import RateLaw
        >>> x, k = var('x'), var('k')
        >>> law = RateLaw(x, -k * x).prepare('decay')
        >>> law.targets_, sorted(map(str, law.reads_)), law.owner_
        ((x,), ['k', 'x'], 'decay')
        >>> law.run({'x': 2.0, 'k': 0.5}).value_
        -1.0
    """

    law_tag: ClassVar[str] = ''

    value_: object

    @abstractmethod
    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the law changes."""

    @classmethod
    @abstractmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the laws of the class require of a system.

        An example is continuous or discrete time.
        """

    def _get_law_requirements(self) -> RequirementSpec:
        """Return what this law requires of a system.

        Its arguments may narrow it.
        """
        return self._get_requirements()

    def _get_tags(self) -> Tags:
        """Return the tags of the law.

        They are its kind, its randomness, and the times its arguments let it run in.
        """
        times = frozenset(getattr(time, 'time_tag', '') for time in self._get_law_requirements().time) - {''}
        return replace(self._get_law_tags([type(self)]), time=times)

    @classmethod
    def get_requirements(cls) -> RequirementSpec:
        """Return what the laws of the class require of a system before any law exists.

        Returns:
            requirements:
                The classes of time and of space the laws run in, where an empty field accepts any class. A law whose
                arguments narrow them, such as `ArrayUpdateLaw`, returns the widest it can run in.

        Examples:
            >>> from skcomplex.core.components.laws import RuleLaw
            >>> RuleLaw.get_requirements().time[0].__name__
            'DiscreteTime'
        """
        return cls._get_requirements()

    def _run(self, inputs: object) -> None:
        """Set `value_` to the values of the arguments of the law at given symbol values.

        The values are those of the symbols the law reads. A law with a value of its own, such as the rate of a
        `RateLaw`, overrides this hook.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self._get_values(), inputs)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the law itself, the one law it contributes, whatever the namespace."""
        return [self]

    def _get_values(self, include: Collection[str] | None = None, exclude: Collection[str] = ()) -> list[object]:
        """Return the values of the arguments of the law besides its label.

        They are flattened in the order of its constructor. A call gives its arguments, a mapping its keys and values in
        pairs, and a vector, a list, or a tuple its items, at any depth. Every other value is returned as it is.

        Args:
            include:
                The names of the arguments to read, or `None` for every argument.

            exclude:
                The names of the arguments to leave out.

        Returns:
            values:
                The flattened values.
        """
        names = [
            name
            for name in inspect.signature(type(self)).parameters
            if name != 'label' and (include is None or name in include) and name not in exclude
        ]
        stack: list[object] = [getattr(self, name) for name in reversed(names)]
        found: list[object] = []
        while stack:
            value = stack.pop()
            if isinstance(value, Mapping):
                stack += reversed([item for pair in value.items() for item in pair])
            elif isinstance(value, FreeModuleElement | list | tuple):
                stack += reversed(list(value))
            elif isinstance(value, CallSpec):
                stack += reversed(value.arguments or ())
            else:
                found.append(value)
        return found

    @staticmethod
    def _is_function(value: object) -> bool:
        """Check whether a law value is a function or a call, not a symbolic expression.

        A symbolic expression is also callable.
        """
        return isinstance(value, CallSpec) or (
            callable(value) and not isinstance(value, Expression | FreeModuleElement | Matrix | numbers.Number)
        )

    def _evaluate(self, value: object, inputs: object) -> object:
        """Return a value with every symbol substituted by its input.

        It is a number when nothing symbolic is left.
        """
        values = inputs if isinstance(inputs, Mapping) else {}
        if isinstance(value, tuple | list):
            return type(value)(self._evaluate(item, values) for item in value)
        if isinstance(value, Mapping):
            return {key: self._evaluate(item, values) for key, item in value.items()}
        if not isinstance(value, Expression):
            return value
        result = value.subs(**values)
        return float(result) if not result.variables() and result.is_real() else result

    def _check_array(self, name: str) -> None:
        """Raise an error for an argument that is a NumPy array.

        NumPy builds one from a scalar times a vector.
        """
        if isinstance(getattr(self, name), np.ndarray):
            error_msg = (
                f'The {name} of {type(self).__name__} is a NumPy array, which NumPy builds when a NumPy scalar '
                'multiplies a symbolic vector. Convert the NumPy scalar with `float(...)` first.'
            )
            raise ValidationError(error_msg)

    def _check_matrix(self, name: str) -> None:
        """Raise an error for an argument that is not a nonempty matrix.

        A matrix is a sequence of rows of equal length.
        """
        matrix = getattr(self, name)
        rows = [list(row) for row in matrix] if isinstance(matrix, list | tuple | np.ndarray) else []
        widths = {len(row) for row in rows}
        if not rows or len(widths) != 1 or 0 in widths:
            error_msg = (
                f'The {name} of {type(self).__name__} must be a nonempty sequence of rows of equal length, but got '
                f'{matrix!r}.'
            )
            raise ValidationError(error_msg)

    def _check_table(self, name: str, dimensions: tuple[int, ...]) -> None:
        """Raise an error for an argument that is not an array of numbers or expressions.

        The array must have one of some dimensions.
        """
        table = np.asarray(getattr(self, name), dtype=object)
        if table.ndim not in dimensions:
            expected = ' or '.join(str(item) for item in dimensions)
            error_msg = (
                f'The {name} of {type(self).__name__} must be an array of {expected} dimensions, but got shape '
                f'{table.shape}.'
            )
            raise ValidationError(error_msg)

    def _check_outcomes(self, rate_name: str, name: str) -> None:
        """Raise an error for an outcome table whose leading shape is not the rate table's.

        Its first two dimensions must be the shape of the rate table.
        """
        rate, outcomes = np.asarray(getattr(self, rate_name), dtype=object), getattr(self, name)
        if outcomes is None or rate.ndim == 0 or rate.shape == np.shape(outcomes)[:2]:
            return
        error_msg = (
            f'The {rate_name} of {type(self).__name__} has the shape {rate.shape}, but its {name} have '
            f'{np.shape(outcomes)[:2]} candidate and field classes.'
        )
        raise ValidationError(error_msg)

    def _prepare(self, context: str | None = None) -> None:
        """Set `targets_`, `reads_`, and `owner_`.

        They are the states the law changes, the symbols it reads, and its owner.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        self.targets_ = self._get_targets()
        self.reads_ = frozenset(
            item for value in self._get_values() if isinstance(value, Expression) for item in value.variables()
        )
        self.owner_ = context
