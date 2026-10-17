"""Lower symbolic and callable law values to fast numeric functions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import sympy

from ..core import ValidationError
from ..model import NumericFunction
from ._layout import Layout
from ._problem import Vector, VectorFunction


@dataclass(frozen=True)
class Context:
    """Symbols and layouts that numeric functions are built against.

    Args:
        time:
            The time symbol.

        state_symbols:
            The symbols of the state components, in vector order.

        parameter_symbols:
            The symbols of the parameter components, in vector order.

        states:
            The layout of the state vector.

        parameters:
            The layout of the parameter vector.
    """

    time: sympy.Symbol
    state_symbols: list[sympy.Symbol]
    parameter_symbols: list[sympy.Symbol]
    states: Layout
    parameters: Layout

    @property
    def known_symbols(self) -> set[sympy.Symbol]:
        """Symbols a law expression may contain."""
        return {self.time, *self.state_symbols, *self.parameter_symbols}

    def values_of(self, x: Vector, p: Vector) -> dict[str, Any]:
        """Return the values of every state and parameter by dotted name.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            values:
                A mapping from dotted name to value.
        """
        return {**self.states.values(x), **self.parameters.values(p)}


@dataclass
class Slots:
    """Symbolic expressions and callables that fill a vector of a given size.

    Args:
        size:
            The size of the vector.

        expressions:
            A mapping from slot index to the sum of its symbolic terms.

        callables:
            Callables with the slot indices their results are added to.
    """

    size: int
    expressions: dict[int, sympy.Expr] = field(default_factory=dict)
    callables: list[tuple[npt.NDArray[np.intp], NumericFunction]] = field(default_factory=list)

    def add_expression(self, index: int, expression: sympy.Expr) -> None:
        """Add a symbolic term to a slot.

        Args:
            index:
                The slot index.

            expression:
                The term.
        """
        self.expressions[index] = self.expressions.get(index, sympy.Integer(0)) + expression

    def add_callable(self, indices: list[int], function: NumericFunction) -> None:
        """Add a callable whose result is added to some slots.

        Args:
            indices:
                The slot indices.

            function:
                The callable `f(t, values)`.
        """
        self.callables.append((np.asarray(indices, dtype=np.intp), function))

    @property
    def callable_indices(self) -> set[int]:
        """Slot indices that receive a callable result."""
        return {int(index) for indices, _ in self.callables for index in indices}


def _check_symbols(expressions: list[sympy.Expr], context: Context, owner: str) -> None:
    """Raise an error for an expression that contains an undeclared symbol."""
    unknown = set().union(*(expression.free_symbols for expression in expressions)) - context.known_symbols
    if unknown:
        names = ', '.join(sorted(map(str, unknown)))
        error_msg = (
            f'A law of {owner} uses the symbols {names}, which are not declared. Declare them with `state` or '
            '`parameter` and use them through the namespace `s` that `laws` receives.'
        )
        raise ValidationError(error_msg)


def _to_expression(component: object) -> sympy.Expr:
    """Return a component as a SymPy expression, converting twice for objects whose `_sympy_` gives a number."""
    expression = sympy.sympify(component)
    return expression if isinstance(expression, sympy.Basic) else sympy.sympify(expression)


def _to_components(value: object) -> list[object] | None:
    """Return the components of a symbolic value, or `None` for a callable."""
    if not isinstance(value, sympy.Basic | sympy.MatrixBase) and hasattr(value, '_sympy_'):
        value = sympy.sympify(value)
    if isinstance(value, sympy.MatrixBase):
        return list(value)
    if isinstance(value, sympy.Basic | numbers.Number):
        return [value]
    if isinstance(value, list | tuple | np.ndarray):
        return list(np.ravel(np.asarray(value, dtype=object)))
    if callable(value):
        return None
    error_msg = f'Law value {value!r} is neither a symbolic expression, a number, a sequence, nor a callable.'
    raise ValidationError(error_msg)


def symbolic_components(value: object, size: int, context: Context, owner: str) -> list[sympy.Expr] | None:
    """Return the symbolic components of a law value, or `None` when the value is a callable.

    Args:
        value:
            The law value.

        size:
            The number of components of the law target. A scalar value is broadcast to every component.

        context:
            The symbols the value may contain.

        owner:
            The path of the component that returned the law, used in error messages.

    Returns:
        components:
            The symbolic components, or `None` for a callable.

    Raises:
        ValidationError:
            The value has the wrong size or contains undeclared symbols.
    """
    components = _to_components(value)
    if components is None:
        return None
    expressions = [_to_expression(component) for component in components]
    if len(expressions) == 1 and size > 1:
        expressions *= size
    if len(expressions) != size:
        error_msg = f'A law of {owner} has a value with {len(expressions)} components for a target with {size}.'
        raise ValidationError(error_msg)
    _check_symbols(expressions, context, owner)
    return expressions


def build_function(slots: Slots, context: Context) -> VectorFunction:
    """Return the numeric function that fills a vector from its slots.

    Args:
        slots:
            The symbolic expressions and callables of the vector.

        context:
            The symbols and layouts the function is evaluated against.

    Returns:
        function:
            The function `f(t, x, p)` that returns the vector.
    """
    indices = np.asarray(sorted(slots.expressions), dtype=np.intp)
    expressions = [slots.expressions[int(index)] for index in indices]
    symbolic: Callable[..., Any] = sympy.lambdify(
        (context.time, context.state_symbols, context.parameter_symbols),
        expressions,
        modules='numpy',
        cse=True,
    )
    callables = slots.callables

    def evaluate(t: float, x: Vector, p: Vector) -> Vector:
        """Evaluate the symbolic expressions and add the callable results."""
        output = np.zeros(slots.size)
        if expressions:
            output[indices] = np.asarray(symbolic(t, x, p), dtype=float).reshape(len(expressions))
        if callables:
            values: Mapping[str, Any] = context.values_of(x, p)
            for callable_indices, function in callables:
                result = np.asarray(function(t, values), dtype=float).reshape(len(callable_indices))
                output[callable_indices] += result
        return output

    return evaluate
