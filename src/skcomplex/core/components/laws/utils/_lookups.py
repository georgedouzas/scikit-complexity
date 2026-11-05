"""Define lookup tables, piecewise linear functions of a value given by points."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression

from ._signals import series


def lookup(value: object, points: Sequence[Sequence[float]]) -> Expression:
    """Return a table function, the piecewise-linear interpolation of a value between points, flat outside them.

    Args:
        value:
            The symbolic input of the table.

        points:
            The points `(x, y)` of the table, with increasing `x`.

    Returns:
        output:
            The symbolic output, nested conditions that every backend lowers.

    Raises:
        ValueError:
            The points are empty or their inputs do not increase.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import lookup
        >>> food = var('food')
        >>> fertility = lookup(food, [(0, 0.0), (1, 2.0), (2, 3.0)])
        >>> [float(fertility.subs({food: value})) for value in (-1, 0.5, 1.5, 5)]
        [0.0, 1.0, 2.5, 3.0]
    """
    table = [(float(x), float(y)) for x, y in points]
    return series(SR(value), [x for x, _ in table], [y for _, y in table])
