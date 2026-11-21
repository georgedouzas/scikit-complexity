"""Define the abstract base of every law."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import inspect
import numbers
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix

from ...base import BaseSpec
from ...errors import ValidationError
from ._call_spec import CallSpec
from ._requirement_spec import RequirementSpec


@dataclass(frozen=True)
class BaseLaw(BaseSpec, ABC):
    """An abstract law, one equation or rule that changes the values of elements.

    A law says how a value changes, such as `RateLaw(s.ball.x.diff(s.t, 2), -s.g)` inside the interaction of a falling
    ball, which gives the height `x` the acceleration `-g`. It is a frozen value, equal to any law of its class with the
    same fields, and it is not added to a system on its own. An interaction returns its laws in `laws_` when it is
    prepared, and the compiler checks them and turns them into a problem that a solver runs. A subclass says which
    values it changes, whether it needs continuous or discrete time, and its value at given values of the symbols it
    reads. Its subclasses give the spread of an epidemic, the next generation of a cellular automaton, a chemical
    reaction one molecule at a time, and the currents of a circuit.

    Read more in the user guide, under Laws.

    See Also:
        RateLaw: Adds a rate to a derivative of a value.
        UpdateLaw: Gives the next value after one step of discrete time.
        Interaction: Returns the laws of the elements its roles bind.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import BaseLaw, RateLaw
        >>> n, r = var('n r')
        >>> isinstance(RateLaw(n, r * n), BaseLaw), RateLaw(n, r * n) == RateLaw(n, r * n)
        (True, True)
    """

    law_tag: ClassVar[str] = ''

    @abstractmethod
    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the law changes."""

    @classmethod
    @abstractmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the laws of the class require of a system.

        An example is continuous or discrete time.
        """

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return cls.__name__

    def _get_law_requirements(self) -> RequirementSpec:
        """Return what this law requires of a system.

        Its arguments may narrow it.
        """
        return self._get_requirements()

    @classmethod
    def get_requirements(cls) -> RequirementSpec:
        """Return what the laws of the class require of a system before any law exists.

        Returns:
            requirements:
                The classes of time and of space the laws run in, where an empty field accepts any class. A law whose
                arguments narrow them, such as an `UpdateLaw` of a whole array, returns the widest it can run in.

        Examples:
            >>> from skcomplex.core.components.laws import UpdateLaw
            >>> UpdateLaw.get_requirements().space
            ()
        """
        return cls._get_requirements()

    def _get_value(self, inputs: object) -> object:
        """Return the values of the arguments of the law at given symbol values.

        The values are those of the symbols the law reads. A law with a value of its own, such as the rate of a
        `RateLaw`, overrides this hook.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        return self._evaluate(self._get_values(), inputs)

    def _get_fields(self) -> dict[str, Any]:
        """Return the value of every field of the law by name, shared, not copied."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

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
                f'The {name} of {self._get_name()} is a NumPy array, which NumPy builds when a NumPy scalar '
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
                f'The {name} of {self._get_name()} must be a nonempty sequence of rows of equal length, but got '
                f'{matrix!r}.'
            )
            raise ValidationError(error_msg)

    def _get_reads(self) -> frozenset[Expression]:
        """Return the symbols the law reads, those of its expressions at any depth."""
        return frozenset(
            item for value in self._get_values() if isinstance(value, Expression) for item in value.variables()
        )
