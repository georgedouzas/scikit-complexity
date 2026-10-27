"""Define the differential operators of fields, as Sage symbolic functions the compiler lowers.

Sage differentiates with respect to symbols, not along the axes of a lattice or a mesh, so each operator is a Sage
function made by `function`, kept unevaluated in expressions until the discretization of the field turns it into
numbers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression, function, vector
from sage.modules.free_module_element import FreeModuleElement

laplacian = function('laplacian', nargs=1, latex_name=r'\nabla^2')
partial = function('partial', nargs=2)
upwind = function('upwind', nargs=3)
inverse_laplacian = function('inverse_laplacian', nargs=1)


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
