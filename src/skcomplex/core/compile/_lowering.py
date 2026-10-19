"""Convert law values to symbolic components or callables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers

import numpy as np
from sage.all__sagemath_symbolics import SR

from ..core import Expression, SymbolicVector, ValidationError


def _is_sympy(value: object) -> bool:
    """Check whether a value comes from SymPy, without importing it."""
    return type(value).__module__.split('.')[0] == 'sympy'


def _sympy_error(value: object, owner: str) -> ValidationError:
    """Return the error of a SymPy value in a law."""
    error_msg = (
        f'A {owner} has the SymPy value {value}. Sage is the symbolic layer, so write the expression with the '
        'symbols of the namespace `s` and Sage functions, or convert it with `expression._sage_()`.'
    )
    return ValidationError(error_msg)


def _to_expression(component: object, owner: str) -> Expression:
    """Return a component as a Sage expression."""
    if _is_sympy(component):
        raise _sympy_error(component, owner)
    if isinstance(component, Expression):
        return component
    if isinstance(component, numbers.Number | np.number):
        return SR(float(component) if isinstance(component, np.floating) else component)
    error_msg = f'A {owner} has the component {component!r}, which is neither a Sage expression nor a number.'
    raise ValidationError(error_msg)


def _to_components(value: object, owner: str) -> list[object] | None:
    """Return the components of a symbolic value, or `None` for a callable."""
    if _is_sympy(value):
        raise _sympy_error(value, owner)
    if isinstance(value, SymbolicVector):
        return list(value)
    if isinstance(value, Expression | numbers.Number | np.number):
        return [value]
    if isinstance(value, list | tuple | np.ndarray):
        return list(np.ravel(np.asarray(value, dtype=object)))
    if callable(value):
        return None
    error_msg = (
        f'A {owner} has the value {value!r}, which is neither a Sage expression, a number, a sequence, nor a callable.'
    )
    raise ValidationError(error_msg)


def symbolic_components(value: object, size: int, owner: str) -> list[Expression] | None:
    """Return the symbolic components of a law value, or `None` when the value is a callable.

    Args:
        value:
            The law value.

        size:
            The number of components of the law target. A scalar value is broadcast to every component.

        owner:
            The law and the path of the component that returned it, such as `Rate law of hunt`, used in error messages.

    Returns:
        components:
            The symbolic components, or `None` for a callable.

    Raises:
        ValidationError:
            The value is a SymPy expression, is of an unknown type, or has the wrong size.
    """
    components = _to_components(value, owner)
    if components is None:
        return None
    expressions = [_to_expression(component, owner) for component in components]
    if len(expressions) == 1 and size > 1:
        expressions *= size
    if len(expressions) != size:
        error_msg = f'A {owner} has a value with {len(expressions)} components for a target with {size}.'
        raise ValidationError(error_msg)
    return expressions
