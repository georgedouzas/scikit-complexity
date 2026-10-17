"""Convert law values to symbolic components or callables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers

import numpy as np
import sympy

from ..core import ValidationError


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


def symbolic_components(value: object, size: int, owner: str) -> list[sympy.Expr] | None:
    """Return the symbolic components of a law value, or `None` when the value is a callable.

    Args:
        value:
            The law value.

        size:
            The number of components of the law target. A scalar value is broadcast to every component.

        owner:
            The path of the component that returned the law, used in error messages.

    Returns:
        components:
            The symbolic components, or `None` for a callable.

    Raises:
        ValidationError:
            The value has the wrong size.
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
    return expressions
