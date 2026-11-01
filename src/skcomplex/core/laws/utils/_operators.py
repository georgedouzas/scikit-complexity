"""Define the differential operators of fields, as Sage symbolic functions the compiler lowers.

Sage differentiates with respect to symbols, not along the axes of a lattice or a mesh, so each operator is a Sage
function made by `function`, kept unevaluated in expressions until the discretization of the field turns it into
numbers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression, function, oo, vector
from sage.modules.free_module_element import FreeModuleElement

from ....errors import ValidationError

laplacian = function('laplacian', nargs=1, latex_name=r'\nabla^2')
partial = function('partial', nargs=2)
upwind = function('upwind', nargs=3)
INVERSE_HELMHOLTZ = function('inverse_helmholtz')
CONVOLVE = function('convolve', nargs=3)
SHIFTED = function('shifted')
INVERSION_BOUNDARIES = ('periodic', 'zero-flux', 'fixed')
ORDERS = (1, 3)


def gradient(expression: object, n_dim: int = 2) -> FreeModuleElement:
    """Return the gradient of a field expression, the derivative along every dimension of a lattice.

    Args:
        expression:
            An expression of the states and parameters of a field.

        n_dim:
            The number of dimensions of the lattice.

    Returns:
        gradient:
            One derivative per dimension.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import gradient
        >>> gradient(var('u'))
        (partial(u, 0), partial(u, 1))
    """
    return vector(SR, [partial(SR(expression), SR(axis)) for axis in range(n_dim)])


def divergence(components: Sequence[object] | FreeModuleElement) -> Expression:
    """Return the divergence of a vector of field expressions, the sum of the derivative of each along its dimension.

    Args:
        components:
            One expression per dimension, such as a flux.

    Returns:
        divergence:
            The divergence.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import divergence
        >>> u = var('u')
        >>> sorted(str(term) for term in divergence([u, 2 * u]).operands())
        ['partial(2*u, 1)', 'partial(u, 0)']
    """
    return sum((partial(SR(item), SR(axis)) for axis, item in enumerate(components)), SR(0))


def advection(expression: object, velocity: Sequence[object] | FreeModuleElement) -> Expression:
    """Return the advection of a field expression by a velocity, its rate of transport, with upwind differences.

    Args:
        expression:
            An expression of the states and parameters of a field.

        velocity:
            One velocity component per dimension, numbers or expressions.

    Returns:
        advection:
            The advection term, to subtract from the rate of the field.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import advection
        >>> sorted(str(term) for term in advection(var('u'), [1.0, 2.0]).operands())
        ['upwind(u, 0, 1.0)', 'upwind(u, 1, 2.0)']
    """
    return sum((upwind(SR(expression), SR(axis), SR(item)) for axis, item in enumerate(velocity)), SR(0))


def inverse_helmholtz(expression: object, screening: object, boundary: object = None) -> Expression:
    """Return the solution `psi` of the modified-Helmholtz equation `(laplacian - screening**2) psi = expression`.

    Args:
        expression:
            An expression of the states and parameters of a field.

        screening:
            The screening, a non-negative expression of the parameters.

        boundary:
            The boundary condition of the solution, `'periodic'`, `'zero-flux'`, or `('fixed', value)`, or `None` for
            the boundary condition of the states of the expression.

    Returns:
        solution:
            The call `inverse_helmholtz(expression, screening)`, followed for a given boundary by its position in
            `INVERSION_BOUNDARIES` and its fixed value, 0 for the other boundaries.

    Raises:
        ValidationError:
            The boundary is not one of `INVERSION_BOUNDARIES` or a fixed boundary with its value.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import inverse_helmholtz
        >>> inverse_helmholtz(var('q'), 2)
        inverse_helmholtz(q, 2)
        >>> inverse_helmholtz(var('q'), 2, ('fixed', 0.5))
        inverse_helmholtz(q, 2, 2, 0.5)
    """
    if boundary is None:
        return INVERSE_HELMHOLTZ(SR(expression), SR(screening))
    kind, *values = boundary if isinstance(boundary, tuple) else (boundary,)
    if kind not in INVERSION_BOUNDARIES or (kind == 'fixed') != bool(values) or len(values) > 1:
        error_msg = (
            f"The boundary of the inversion is {boundary!r}. Use 'periodic', 'zero-flux', or ('fixed', value), or None."
        )
        raise ValidationError(error_msg)
    code = SR(INVERSION_BOUNDARIES.index(kind))
    return INVERSE_HELMHOLTZ(SR(expression), SR(screening), code, SR(values[0] if values else 0))


def inverse_laplacian(expression: object) -> Expression:
    """Return the solution of the Poisson equation `laplacian(psi) = expression`, the unscreened inverse Helmholtz.

    Args:
        expression:
            An expression of the states and parameters of a field.

    Returns:
        solution:
            The call `inverse_helmholtz(expression, 0)`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import inverse_laplacian
        >>> inverse_laplacian(var('q'))
        inverse_helmholtz(q, 0)
    """
    return INVERSE_HELMHOLTZ(SR(expression), SR(0))


def displacement(axis: int) -> Expression:
    """Return the symbol of the displacement along an axis, the variable a convolution kernel is written in.

    Args:
        axis:
            The axis of the lattice.

    Returns:
        displacement:
            The real symbol `displacement_<axis>`.

    Examples:
        >>> from skcomplex.core.laws.utils import displacement
        >>> displacement(0) ** 2
        displacement_0^2
    """
    return SR.var(f'displacement_{axis}', domain='real')


def convolve(expression: object, kernel: object, radius: object = None) -> Expression:
    """Return the convolution of a field expression with a kernel of the displacement, the sum over sites times volume.

    Args:
        expression:
            An expression of the states and parameters of a field on a lattice.

        kernel:
            An expression of the symbols `displacement(axis)` and of parameters.

        radius:
            The distance beyond which the kernel is zero, or `None` for a kernel over the whole lattice.

    Returns:
        convolution:
            The call `convolve(expression, kernel, radius)`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var, exp
        >>> from skcomplex.core.laws.utils import convolve, displacement
        >>> convolve(var('u'), exp(-displacement(0) ** 2), 3)
        convolve(u, e^(-displacement_0^2), 3)
    """
    return CONVOLVE(SR(expression), SR(kernel), oo if radius is None else SR(radius))


def shifted(expression: object, offset: Sequence[object], order: int = 1) -> Expression:
    """Return a field expression read at a physical displacement from every site, one component per axis.

    Args:
        expression:
            An expression of the states and parameters of a field on a lattice.

        offset:
            The displacement along every axis, numbers or expressions of the states and parameters.

        order:
            The order of the spline that interpolates between sites, 1 for linear or 3 for cubic.

    Returns:
        read:
            The call `shifted(expression, *offset)`, followed by the order when it is 3.

    Raises:
        ValidationError:
            The order is not 1 or 3.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import shifted
        >>> shifted(var('u'), [var('v'), 1])
        shifted(u, v, 1)
        >>> shifted(var('u'), [var('v'), 1], order=3)
        shifted(u, v, 1, 3)
    """
    if order not in ORDERS:
        error_msg = f'The order of a shifted read is {order!r}. Use 1 for linear or 3 for cubic interpolation.'
        raise ValidationError(error_msg)
    orders = [SR(order)] if order != 1 else []
    return SHIFTED(SR(expression), *[SR(item) for item in offset], *orders)
