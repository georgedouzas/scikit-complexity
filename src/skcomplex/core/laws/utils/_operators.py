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

laplacian = function('laplacian', nargs=1, latex_name=r'\nabla^2')
partial = function('partial', nargs=2)
upwind = function('upwind', nargs=3)
inverse_helmholtz = function('inverse_helmholtz', nargs=2)
CONVOLVE = function('convolve', nargs=3)
SHIFTED = function('shifted')


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
    return inverse_helmholtz(SR(expression), SR(0))


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


def shifted(expression: object, offset: Sequence[object]) -> Expression:
    """Return a field expression read at a physical displacement from every site, one component per axis.

    Args:
        expression:
            An expression of the states and parameters of a field on a lattice.

        offset:
            The displacement along every axis, numbers or expressions of the states and parameters.

    Returns:
        read:
            The call `shifted(expression, *offset)`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import shifted
        >>> shifted(var('u'), [var('v'), 1])
        shifted(u, v, 1)
    """
    return SHIFTED(SR(expression), *[SR(item) for item in offset])
