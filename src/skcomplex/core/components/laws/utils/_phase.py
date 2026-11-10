"""Define the coordinates and reductions over the class axes of a phase space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression

from ....errors import ValidationError
from ._operators import _OperatorFunction

COORDINATE_PREFIX = '_coordinate_'
REDUCE_AXES = _OperatorFunction('reduce_axes')
MAXIMUM_ENTROPY = _OperatorFunction('maximum_entropy')


def coordinate(name: str) -> Expression:
    """Return the coordinate of the point a law updates along a named class axis.

    The axis is a class axis of a phase space.

    Args:
        name:
            The name of the class axis.

    Returns:
        coordinate:
            The symbol of the coordinate, a real number at every point of the space.

    Raises:
        ValidationError:
            The name is no identifier.

    Examples:
        >>> from skcomplex.core.components.laws.utils import coordinate
        >>> coordinate('v') ** 2
        _coordinate_v^2
    """
    if not isinstance(name, str) or not name.isidentifier():
        error_msg = f'A class axis is named by an identifier, but it was given {name!r}.'
        raise ValidationError(error_msg)
    return SR.var(f'{COORDINATE_PREFIX}{name}', domain='real')


def coordinates_of(axes: Sequence[str]) -> list[Expression]:
    """Return the coordinates of named class axes.

    Args:
        axes:
            The names of the class axes.

    Returns:
        coordinates:
            The symbol of the coordinate of every axis.

    Raises:
        ValidationError:
            No axis is named, or an axis is named by a value that is no identifier.

    Examples:
        >>> from skcomplex.core.components.laws.utils._phase import coordinates_of
        >>> coordinates_of(('v', 'w'))
        [_coordinate_v, _coordinate_w]
    """
    if isinstance(axes, str) or not axes:
        error_msg = f'Name the class axes by a sequence of names, such as `axes=("v",)`, not {axes!r}.'
        raise ValidationError(error_msg)
    return [coordinate(name) for name in axes]


def reduce_axes(expression: object, axes: Sequence[str]) -> Expression:
    """Return the integral of an expression over named class axes.

    It is taken at every position of a phase space.

    Args:
        expression:
            An expression of the states of a field on a phase space and the coordinates.

        axes:
            The names of the class axes to integrate over, each cell weighted by its width.

    Returns:
        integral:
            The call `reduce_axes(expression, *coordinates)`, which laws read broadcast along the reduced axes.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import reduce_axes
        >>> reduce_axes(var('f'), ('v',))
        reduce_axes(f, _coordinate_v)
    """
    return REDUCE_AXES(SR(expression), *coordinates_of(axes))


def maximum_entropy(expression: object, axes: Sequence[str], order: int = 2) -> Expression:
    """Return the density of largest entropy with moments up to an order of another.

    Its moments up to an order equal those of an expression. The density is `exp(a + b c)` for the order 1 and `exp(a +
    b c + d |c|^2)` for the order 2, with `c` the coordinates of the named axes, and its parameters are found at every
    position so that its discrete moments equal those of the expression to round-off.

    Args:
        expression:
            An expression of the states of a field on a phase space and the coordinates.

        axes:
            The names of the class axes of the density.

        order:
            The highest order of the moments that are kept, 0, 1, or 2.

    Returns:
        density:
            The call `maximum_entropy(expression, order, *coordinates)`.

    Raises:
        ValidationError:
            The order is not 0, 1, or 2.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import maximum_entropy
        >>> maximum_entropy(var('f'), ('v',))
        maximum_entropy(f, 2, _coordinate_v)
    """
    if order not in {0, 1, 2}:
        error_msg = f'The order of a maximum-entropy density is 0, 1, or 2, not {order!r}.'
        raise ValidationError(error_msg)
    return MAXIMUM_ENTROPY(SR(expression), SR(order), *coordinates_of(axes))
