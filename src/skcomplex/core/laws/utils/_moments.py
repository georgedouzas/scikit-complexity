"""Define the moments of a distribution held by a group whose members stand for the points of a grid of coordinates."""

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
    """Return the sum of an expression times the coordinates raised to an order, over the members of a group."""
    if order == 0:
        return total(expression, over=over)
    return total(expression, over=over, weights=coordinates**order)


def _raw_over_axes(expression: object, order: int, axes: Sequence[str]) -> Expression:
    """Return the integral over class axes of an expression times the length of the coordinates to an order."""
    coordinates = coordinates_of(axes)
    if len(coordinates) == 1:
        return reduce_axes(SR(expression) * coordinates[0] ** order, axes)
    if order % 2:
        error_msg = f'A moment over the axes {tuple(axes)} has an even order. Reduce over one axis, not order {order}.'
        raise ValidationError(error_msg)
    return reduce_axes(SR(expression) * sum(item**2 for item in coordinates) ** (order // 2), axes)


def _combined(raw: Callable[[int], Expression], order: int, normalized: bool, central: bool) -> Expression:
    """Return a raw, normalized, or central moment from the function of the raw moment of every order."""
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
    """Return a moment of the distribution that a group holds over a grid of coordinates, one point per member.

    The raw moment of order `r` is `sum_j c_j^r f_j`, where member `j` holds the density `f_j` at the coordinate `c_j`.
    The order-0 moment is the density. A normalized moment divides by the density, so the normalized order-1 moment
    is the mean coordinate. A central moment is normalized and taken about the mean, so the central order-2 moment is
    the variance.

    With `axes`, the moment reduces a field on a phase space over those class axes at every position, by the sum over
    the cells of the coordinate raised to the order times the cell widths, and it is a field over position that laws
    read broadcast along the reduced axes. Over several axes the coordinate is the length of the vector of coordinates.

    Args:
        expression:
            The expression of one member, usually the density state.

        order:
            The order `r`, a nonnegative integer.

        coordinates:
            The coordinate of every member, such as the activity values of a grid, or `None` with `axes`.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        normalized:
            Whether to divide by the density.

        central:
            Whether to take the moment about the mean, which also normalizes it.

        axes:
            The names of the class axes of a phase space to reduce over, or `None` to reduce over every member.

    Returns:
        moment:
            The moment as an expression of reductions, which laws read and observables record.

    Raises:
        ValidationError:
            The order is not a nonnegative integer, the coordinates are not a nonempty vector, both coordinates and
            axes are given, or an odd order is taken over several axes.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import moment, reductions_in
        >>> f = var('f')
        >>> moment(f, 0, [0.0, 1.0, 2.0])
        total(f)
        >>> sorted(item.weights is None for item in reductions_in(moment(f, 1, [0.0, 1.0, 2.0], normalized=True)))
        [False, True]
        >>> moment(f, 2, axes=('v',))
        reduce_axes(_coordinate_v^2*f, _coordinate_v)
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        error_msg = f'The order of a moment is a nonnegative integer, not {order!r}.'
        raise ValidationError(error_msg)
    if axes is not None:
        if coordinates is not None:
            error_msg = f'A moment over the axes {axes} reads the coordinates of the axes. Give no other coordinates.'
            raise ValidationError(error_msg)
        return _combined(lambda power: _raw_over_axes(expression, power, axes), order, normalized, central)
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 1 or not values.size:
        error_msg = f'The coordinates of a moment are a nonempty vector with one value per member, not {coordinates!r}.'
        raise ValidationError(error_msg)
    return _combined(lambda power: _raw(expression, power, values, over), order, normalized, central)
