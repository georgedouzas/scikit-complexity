"""Define the moments of a distribution held by a group whose members stand for the points of a grid of coordinates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import SR

from ..base import Expression, ValidationError
from ._reductions import total


def _coordinates(coordinates: object) -> np.ndarray:
    """Return the coordinates of the members as a float vector, after checking that it is one."""
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 1 or not values.size:
        error_msg = f'The coordinates of a moment are a nonempty vector with one value per member, not {coordinates!r}.'
        raise ValidationError(error_msg)
    return values


def _raw(expression: object, order: int, coordinates: np.ndarray, over: object) -> Expression:
    """Return the sum of an expression times the coordinates raised to an order, over the members of a group."""
    if order == 0:
        return total(expression, over=over)
    return total(expression, over=over, weights=coordinates**order)


def moment(
    expression: object,
    order: int,
    coordinates: object,
    *,
    over: object = None,
    normalized: bool = False,
    central: bool = False,
) -> Expression:
    """Return a moment of the distribution that a group holds over a grid of coordinates, one point per member.

    The raw moment of order `r` is `sum_j c_j^r f_j`, where member `j` holds the density `f_j` at the coordinate `c_j`.
    The order-0 moment is the density. A normalized moment divides by the density, so the normalized order-1 moment
    is the mean coordinate. A central moment is normalized and taken about the mean, so the central order-2 moment is
    the variance.

    Args:
        expression:
            The expression of one member, usually the density state, such as `s.wealth.f`.

        order:
            The order `r`, a nonnegative integer.

        coordinates:
            The coordinate of every member, such as the activity values of a grid.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        normalized:
            Whether to divide by the density.

        central:
            Whether to take the moment about the mean, which also normalizes it.

    Returns:
        moment:
            The moment as an expression of reductions, which laws read and observables record.

    Raises:
        ValidationError:
            The order is not a nonnegative integer, or the coordinates are not a nonempty vector.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import moment, reductions_in
        >>> f = symbol('f')
        >>> moment(f, 0, [0.0, 1.0, 2.0])
        total(f)
        >>> sorted(item.weights is None for item in reductions_in(moment(f, 1, [0.0, 1.0, 2.0], normalized=True)))
        [False, True]
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        error_msg = f'The order of a moment is a nonnegative integer, not {order!r}.'
        raise ValidationError(error_msg)
    values = _coordinates(coordinates)
    if not (normalized or central):
        return _raw(expression, order, values, over)
    density = _raw(expression, 0, values, over)
    if not central:
        return _raw(expression, order, values, over) / density
    mean = _raw(expression, 1, values, over) / density
    terms = [
        math.comb(order, power) * _raw(expression, power, values, over) / density * (-mean) ** (order - power)
        for power in range(order + 1)
    ]
    return SR(sum(terms))
