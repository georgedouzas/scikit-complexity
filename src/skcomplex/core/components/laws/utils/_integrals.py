"""Define the integrals of fields against two-point kernels and along one axis.

The compiler lowers them.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression, function

from ....errors import ValidationError
from ._phase import coordinates_of

INTEGRATION_PREFIX = '_integration_variable_'
INTEGRAL_TRANSFORM = function('integral_transform')
CUMULATIVE_INTEGRAL = function('cumulative_integral', nargs=3)


def integration_variable(name: str) -> Expression:
    """Return the coordinate along a named class axis of an integration point.

    The point is the one an integral transform integrates over.

    Args:
        name:
            The name of the class axis.

    Returns:
        variable:
            The symbol of the coordinate, a real number at every point integrated over.

    Raises:
        ValidationError:
            The name is no identifier.

    Examples:
        >>> from skcomplex.core.components.laws.utils import coordinate, integration_variable
        >>> coordinate('z') - integration_variable('z')
        _coordinate_z - _integration_variable_z
    """
    if not isinstance(name, str) or not name.isidentifier():
        error_msg = f'A class axis is named by an identifier, but it was given {name!r}.'
        raise ValidationError(error_msg)
    return SR.var(f'{INTEGRATION_PREFIX}{name}', domain='real')


def integral_transform(expression: object, kernel: object, axes: Sequence[str]) -> Expression:
    """Return the integral of an expression over named class axes against a kernel.

    The kernel is a kernel of two points. The value at the point `c` is the integral of `K(c, c') u(c')` over the points
    `c'` of the named axes, at the same position and the same coordinates of the other class axes. The kernel is written
    in the coordinates `coordinate(name)` of `c`, the variables `integration_variable(name)` of `c'`, and the
    parameters, so it need not be a function of `c - c'`.

    Args:
        expression:
            An expression of the states of a field on a phase space and the coordinates.

        kernel:
            The kernel, an expression of the coordinates and the integration variables of the axes and the parameters.

        axes:
            The names of the class axes to integrate over, each cell weighted by its width and its volume.

    Returns:
        integral:
            The call `integral_transform(expression, kernel, *coordinates)`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var, tanh
        >>> from skcomplex.core.components.laws.utils import coordinate, integral_transform, integration_variable
        >>> kernel = tanh(coordinate('z') - integration_variable('z') ** 2)
        >>> integral_transform(var('f'), kernel, ('z',))
        integral_transform(f, tanh(-_integration_variable_z^2 + _coordinate_z), _coordinate_z)
    """
    return INTEGRAL_TRANSFORM(SR(expression), SR(kernel), *coordinates_of(axes))


def cumulative_integral(expression: object, axis: int, *, reverse: bool = False) -> Expression:
    """Return the integral of an expression along one axis of a lattice.

    It runs up to every site, or from it to the end. The integral runs over the cells from the start of the domain to
    the site, or from the site to the end of the domain when reversed, and counts half of the cell of the site itself.
    Every cell is weighted by its width and its volume.

    Args:
        expression:
            An expression of the states of a field on a lattice.

        axis:
            The axis of the lattice to integrate along, a class axis of a phase space after its position axes.

        reverse:
            Whether to integrate from every site to the end of the domain rather than from the start to the site.

    Returns:
        integral:
            The call `cumulative_integral(expression, axis, reverse)`.

    Raises:
        ValidationError:
            The axis is not a nonnegative integer.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import cumulative_integral
        >>> cumulative_integral(var('f'), 1, reverse=True)
        cumulative_integral(f, 1, 1)
    """
    if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
        error_msg = f'A cumulative integral runs along an axis given by a nonnegative integer, not {axis!r}.'
        raise ValidationError(error_msg)
    return CUMULATIVE_INTEGRAL(SR(expression), SR(axis), SR(int(reverse)))
