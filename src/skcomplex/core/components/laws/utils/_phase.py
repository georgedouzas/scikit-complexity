"""Define the coordinates of a phase space and the sums over its axes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression

from ....errors import ValidationError
from ._operators import _OperatorFunction
from ._reductions import _calls_in

COORDINATE_PREFIX = '_coordinate_'
REDUCE_AXES = _OperatorFunction('reduce_axes')
MAXIMUM_ENTROPY = _OperatorFunction('maximum_entropy')


def coordinate(name: str) -> Expression:
    """Return the coordinate along a named axis of the point a law updates.

    A phase space adds axes beyond position, such as speed, size, or age, and a law of a field on it reads the
    coordinate of its point along such an axis, such as `coordinate('v') ** 2` for the square of the speed. It weighs
    the energy of particles by speed, the growth of cells by size, and the death rate of people by age.

    Read more in the user guide, under Laws.

    Args:
        name:
            The name of the axis.

    Returns:
        coordinate:
            The symbol of the coordinate, a real number at every point of the space.

    Raises:
        ValidationError:
            The name is not a valid Python name.

    See Also:
        reduce_axes: Integrates over some axes at every position.
        integration_variable: Gives the point an integral runs over.

    Examples:
        >>> from skcomplex.core.components.laws.utils import coordinate
        >>> coordinate('v') ** 2
        _coordinate_v^2
    """
    if not isinstance(name, str) or not name.isidentifier():
        error_msg = f'The name of an axis must be a valid Python name, such as `v`, but it is {name!r}.'
        raise ValidationError(error_msg)
    return SR.var(f'{COORDINATE_PREFIX}{name}', domain='real')


def coordinates_of(axes: Sequence[str]) -> list[Expression]:
    """Return the coordinates of several named axes of a phase space.

    It gives `coordinate(name)` for every name, such as the coordinates of `v` and `w` for `('v', 'w')`, and checks that
    the names come as a sequence, not as a single string. The integrals and moments over axes read it. It gives the
    speeds of particles along two directions, the size and age of cells, and the income and wealth of households.

    Read more in the user guide, under Extend the library.

    Args:
        axes:
            The names of the axes.

    Returns:
        coordinates:
            The symbol of the coordinate of every axis.

    Raises:
        ValidationError:
            No axis is named, or a name is not a valid Python name.

    See Also:
        coordinate: Gives the coordinate of one axis.

    Examples:
        >>> from skcomplex.core.components.laws.utils._phase import coordinates_of
        >>> coordinates_of(('v', 'w'))
        [_coordinate_v, _coordinate_w]
    """
    if isinstance(axes, str) or not axes:
        error_msg = f'Name the axes by a sequence of names, such as `axes=("v",)`, not {axes!r}.'
        raise ValidationError(error_msg)
    return [coordinate(name) for name in axes]


def reduce_axes(expression: object, axes: Sequence[str]) -> Expression:
    """Return the integral of an expression over some axes of a phase space.

    At every position it sums the expression over the cells of the named axes, each weighted by its width, such as
    `reduce_axes(f, ('v',))` for the density of particles of any speed. Laws read the result as the same value at every
    point of the summed axes. It gives the density of a gas from its speeds, the number of cells of all sizes, and the
    population of all ages.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states of a field on a phase space and the coordinates.

        axes:
            The names of the axes to integrate over, each cell weighted by its width.

    Returns:
        integral:
            The call `reduce_axes(expression, *coordinates)`, which laws read as the same value along the summed axes.

    See Also:
        moment: Integrates a power of the coordinates.
        integral_transform: Integrates against a kernel of two points.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import reduce_axes
        >>> reduce_axes(var('f'), ('v',))
        reduce_axes(f, _coordinate_v)
    """
    return REDUCE_AXES(SR(expression), *coordinates_of(axes))


def maximum_entropy(expression: object, axes: Sequence[str], order: int = 2) -> Expression:
    """Return the smoothest density with the same low moments as an expression.

    It is the density of largest entropy whose moments up to `order` equal those of the expression, `exp(a + b c)` for
    the order 1 and `exp(a + b c + d |c|^2)` for the order 2, with `c` the coordinates of the named axes, such as
    `maximum_entropy(f, ('v',))` for the bell curve of speeds with the density, mean, and spread of `f`. Its parameters
    are found at every position so that its moments equal those of the expression up to rounding. It relaxes a gas
    toward equilibrium, closes the equations of a crowd, and fills in a distribution of incomes from its mean and
    variance.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states of a field on a phase space and the coordinates.

        axes:
            The names of the axes of the density.

        order:
            The highest order of the moments that are kept, 0, 1, or 2.

    Returns:
        density:
            The call `maximum_entropy(expression, order, *coordinates)`.

    Raises:
        ValidationError:
            The order is not 0, 1, or 2.

    See Also:
        moment: Gives a moment of a density.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import maximum_entropy
        >>> maximum_entropy(var('f'), ('v',))
        maximum_entropy(f, 2, _coordinate_v)
    """
    if order not in {0, 1, 2}:
        error_msg = f'The order of a maximum entropy density must be 0, 1, or 2, but it is {order!r}.'
        raise ValidationError(error_msg)
    return MAXIMUM_ENTROPY(SR(expression), SR(order), *coordinates_of(axes))


def axis_reductions_in(expression: object) -> list[Expression]:
    """Return the calls of `reduce_axes` in an expression, outermost first.

    The compiler reads them to sum a field over axes of a phase space, each call listed once, such as the call in
    `1 + reduce_axes(f, ('v',))`. A call inside another call is left for the outer one. It finds the densities of gases,
    cell populations, and age structures.

    Read more in the user guide, under Extend the library.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        calls:
            The calls of `reduce_axes`, such as those a moment over axes builds.

    See Also:
        reduce_axes: Builds the calls it finds.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import reduce_axes
        >>> from skcomplex.core.components.laws.utils._phase import axis_reductions_in
        >>> axis_reductions_in(1 + reduce_axes(var('f'), ('v',)))
        [reduce_axes(f, _coordinate_v)]
    """
    return _calls_in(expression, {REDUCE_AXES.name()})
