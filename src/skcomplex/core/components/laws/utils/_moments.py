"""Define the moments of a distribution that a group holds over a grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....errors import ValidationError
from ._phase import coordinates_of, reduce_axes
from ._reductions import total


def _raw(expression: object, order: int, coordinates: np.ndarray, over: object) -> Expression:
    """Return the sum over the members of a group of an expression times coordinates.

    The coordinates are raised to an order.
    """
    if order == 0:
        return total(expression, over=over)
    return total(expression, over=over, weights=coordinates**order)


def _raw_over_axes(expression: object, order: int, axes: Sequence[str]) -> Expression:
    """Return the integral of an expression times a coordinate power over some axes.

    The power is the length of the coordinates to an order.
    """
    coordinates = coordinates_of(axes)
    if len(coordinates) == 1:
        return reduce_axes(SR(expression) * coordinates[0] ** order, axes)
    if order % 2:
        error_msg = (
            f'A moment over several axes, here {tuple(axes)}, must have an even order, but it is {order}. Take an odd '
            'order over one axis.'
        )
        raise ValidationError(error_msg)
    return reduce_axes(SR(expression) * sum(item**2 for item in coordinates) ** (order // 2), axes)


def _combined(raw: Callable[[int], Expression], order: int, normalized: bool, central: bool) -> Expression:
    """Return a raw, normalized, or central moment.

    It comes from the function of the raw moment of every order.
    """
    if not (normalized or central):
        return raw(order)
    density = raw(0)
    if not central:
        return raw(order) / density
    mean = raw(1) / density
    terms = [math.comb(order, power) * raw(power) / density * (-mean) ** (order - power) for power in range(order + 1)]
    return SR(sum(terms))


def moment(
    expression: object,
    order: int,
    coordinates: object = None,
    *,
    over: object = None,
    normalized: bool = False,
    central: bool = False,
    axes: Sequence[str] | None = None,
) -> Expression:
    """Return a moment of a distribution that a group holds over a grid.

    Every member `j` holds the density `f_j` at the coordinate `c_j`, and the moment of order `r` is `Σ_j c_j^r f_j`,
    such as `moment(f, 0, [0.0, 1.0, 2.0])` that gives the total `total(f)`. A normalized moment divides by the total,
    so the normalized moment of order 1 is the mean coordinate, and a central moment is taken about the mean, so the
    central moment of order 2 is the variance. With `axes`, it sums a field of a phase space over those axes, such as
    velocity, at every position, and over several axes the coordinate is the length of the vector of coordinates. It
    gives the mean activity of a neural population, the mean wealth of a class of households, and the pressure of a gas.

    Read more in the user guide, under Laws.

    Args:
        expression:
            The expression of one member, usually the density.

        order:
            The order `r`, a whole number of 0 or more.

        coordinates:
            The coordinate of every member, such as the activity values of a grid, or `None` with `axes`.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        normalized:
            Whether to divide by the density.

        central:
            Whether to take the moment about the mean, which also normalizes it.

        axes:
            The names of the axes of a phase space to sum over, or `None` to sum over every member.

    Returns:
        moment:
            The moment as an expression of sums, which laws read and observables record.

    Raises:
        ValidationError:
            The order is not a whole number of 0 or more, the coordinates are not a list of one number per member, both
            coordinates and axes are given, or an odd order is taken over several axes.

    See Also:
        total: Sums an expression over the members of a group.
        reduce_axes: Sums a field of a phase space over some axes.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import moment
        >>> moment(var('f'), 0, [0.0, 1.0, 2.0])
        total(f)
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        error_msg = f'The order of a moment must be a whole number of 0 or more, but it is {order!r}.'
        raise ValidationError(error_msg)
    if axes is not None:
        if coordinates is not None:
            error_msg = f'A moment over the axes {axes} reads the coordinates of the axes. Give no other coordinates.'
            raise ValidationError(error_msg)
        return _combined(lambda power: _raw_over_axes(expression, power, axes), order, normalized, central)
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 1 or not values.size:
        error_msg = f'The coordinates of a moment must be one number per member, but they are {coordinates!r}.'
        raise ValidationError(error_msg)
    return _combined(lambda power: _raw(expression, power, values, over), order, normalized, central)
