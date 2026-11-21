"""Define the integrals of fields against two-point kernels and along one axis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression

from ....errors import ValidationError
from ._operators import _OperatorFunction
from ._phase import coordinates_of

INTEGRATION_PREFIX = '_integration_variable_'
INTEGRAL_TRANSFORM = _OperatorFunction('integral_transform')
CUMULATIVE_INTEGRAL = _OperatorFunction('cumulative_integral', nargs=3)


def integration_variable(name: str) -> Expression:
    """Return the coordinate along a named axis of the point an integral runs over.

    Inside the kernel of `integral_transform`, `coordinate(name)` is the point that receives the integral and
    `integration_variable(name)` the point it sums over, such as `coordinate('z') - integration_variable('z')` for the
    gap between the two. The axis is one of the axes of a phase space beyond position, such as size, age, or speed. It
    writes how a fragment of one size breaks into smaller ones, how parents of one age give birth, and how particles of
    one speed scatter to another.

    Read more in the user guide, under Laws.

    Args:
        name:
            The name of the axis.

    Returns:
        variable:
            The symbol of the coordinate, a real number at every point integrated over.

    Raises:
        ValidationError:
            The name is not a valid Python name.

    See Also:
        coordinate: Gives the coordinate of the point that receives the integral.
        integral_transform: Integrates against a kernel of both points.

    Examples:
        >>> from skcomplex.core.components.laws.utils import coordinate, integration_variable
        >>> coordinate('z') - integration_variable('z')
        _coordinate_z - _integration_variable_z
    """
    if not isinstance(name, str) or not name.isidentifier():
        error_msg = f'The name of an axis must be a valid Python name, such as `v`, but it is {name!r}.'
        raise ValidationError(error_msg)
    return SR.var(f'{INTEGRATION_PREFIX}{name}', domain='real')


def integral_transform(expression: object, kernel: object, axes: Sequence[str]) -> Expression:
    """Return the integral of an expression over some axes, weighted by a kernel.

    The value at the point `c` is the integral of `K(c, c') u(c')` over the points `c'` of the named axes, at the same
    position and the same coordinates of the other axes. The kernel `K` is written in `coordinate(name)` for `c`,
    `integration_variable(name)` for `c'`, and the parameters, so it need not depend on `c - c'` alone, such as
    `tanh(coordinate('z') - integration_variable('z') ** 2)`. It gives the fragments that break off larger ones, the
    newborns of parents of every age, and the particles that scatter from one speed to another.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states of a field on a phase space and the coordinates.

        kernel:
            The kernel, an expression of the coordinates, the integration variables, and the parameters.

        axes:
            The names of the axes to integrate over, each cell weighted by its width and its volume.

    Returns:
        integral:
            The call `integral_transform(expression, kernel, *coordinates)`.

    See Also:
        cumulative_integral: Integrates along one axis up to every site.
        convolve: Integrates against a kernel of the gap between two points.

    Examples:
        >>> from sage.all__sagemath_symbolics import var, tanh
        >>> from skcomplex.core.components.laws.utils import coordinate, integral_transform, integration_variable
        >>> kernel = tanh(coordinate('z') - integration_variable('z') ** 2)
        >>> integral_transform(var('f'), kernel, ('z',))
        integral_transform(f, tanh(-_integration_variable_z^2 + _coordinate_z), _coordinate_z)
    """
    return INTEGRAL_TRANSFORM(SR(expression), SR(kernel), *coordinates_of(axes))


def cumulative_integral(expression: object, axis: int, *, reverse: bool = False) -> Expression:
    """Return the running integral of an expression along one axis of a lattice.

    At every site it sums the cells from the start of the axis to the site, or from the site to the end with
    `reverse=True`, counting half of the cell of the site itself and weighting every cell by its width and its volume,
    such as `cumulative_integral(f, 1, reverse=True)` for the amount above every size along axis 1. It gives the mass of
    all larger particles, the number of older people in a population, and the wealth of the richer households.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states of a field on a lattice.

        axis:
            The axis of the lattice to integrate along, such as an axis of a phase space after its position axes.

        reverse:
            Whether to integrate from every site to the end of the axis rather than from the start to the site.

    Returns:
        integral:
            The call `cumulative_integral(expression, axis, reverse)`.

    Raises:
        ValidationError:
            The axis is not a whole number of 0 or more.

    See Also:
        integral_transform: Integrates over some axes against a kernel.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import cumulative_integral
        >>> cumulative_integral(var('f'), 1, reverse=True)
        cumulative_integral(f, 1, 1)
    """
    if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
        error_msg = f'The axis of a cumulative integral must be a whole number of 0 or more, but it is {axis!r}.'
        raise ValidationError(error_msg)
    return CUMULATIVE_INTEGRAL(SR(expression), SR(axis), SR(int(reverse)))
