"""Define the differential operators of fields as Sage symbolic functions.

The compiler lowers them. Sage differentiates with respect to symbols, not along the axes of a lattice or a mesh, so
each operator is an instance of a Sage `BuiltinFunction` subclass, kept unevaluated in expressions until the
discretization of the field turns it into numbers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Self

from sage.all__sagemath_symbolics import SR, Expression, oo, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.symbolic.function import BuiltinFunction

from ....errors import ValidationError

COLLISION_SPEED = SR.symbol('collision_speed', domain='real')
COLLISION_COSINE = SR.symbol('collision_cosine', domain='real')
INVERSION_BOUNDARIES = ('periodic', 'zero-flux', 'fixed')
ORDERS = (1, 3)


class _OperatorFunction(BuiltinFunction):
    """Operator kept unevaluated in expressions until the compiler lowers it.

    The name, the number of operands, 0 for any number, and the LaTeX name are those of Sage's `BuiltinFunction`. Sage
    pickles a `BuiltinFunction` by calling its class with no argument, so the operator pickles with the arguments it
    was built with instead, which lets expressions that hold it reach other processes.
    """

    def __init__(self, name: str, nargs: int = 0, latex_name: str | None = None) -> None:
        super().__init__(name, nargs=nargs, latex_name=latex_name)
        self._arguments = (name, nargs, latex_name)

    def __reduce__(self) -> tuple[type[Self], tuple[str, int, str | None]]:
        return type(self), self._arguments


laplacian = _OperatorFunction('laplacian', nargs=1, latex_name=r'\nabla^2')
partial = _OperatorFunction('partial', nargs=2)
upwind = _OperatorFunction('upwind', nargs=3)
COLLISION_INTEGRAL = _OperatorFunction('collision_integral', nargs=2)
INVERSE_HELMHOLTZ = _OperatorFunction('inverse_helmholtz')
CONVOLVE = _OperatorFunction('convolve', nargs=3)
SHIFTED = _OperatorFunction('shifted')


def gradient(expression: object, n_dim: int = 2) -> FreeModuleElement:
    """Return the gradient of a field expression.

    It is the derivative along every dimension of a lattice.

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
        >>> from skcomplex.core.components.laws.utils import gradient
        >>> gradient(var('u'))
        (partial(u, 0), partial(u, 1))
    """
    return vector(SR, [partial(SR(expression), SR(axis)) for axis in range(n_dim)])


def divergence(components: Sequence[object] | FreeModuleElement) -> Expression:
    """Return the divergence of a vector of field expressions.

    It is the sum of the derivative of each along its dimension.

    Args:
        components:
            One expression per dimension, such as a flux.

    Returns:
        divergence:
            The divergence.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import divergence
        >>> u = var('u')
        >>> sorted(str(term) for term in divergence([u, 2 * u]).operands())
        ['partial(2*u, 1)', 'partial(u, 0)']
    """
    return sum((partial(SR(item), SR(axis)) for axis, item in enumerate(components)), SR(0))


def advection(expression: object, velocity: Sequence[object] | FreeModuleElement) -> Expression:
    """Return the advection of a field expression by a velocity.

    It is its rate of transport, with upwind differences.

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
        >>> from skcomplex.core.components.laws.utils import advection
        >>> sorted(str(term) for term in advection(var('u'), [1.0, 2.0]).operands())
        ['upwind(u, 0, 1.0)', 'upwind(u, 1, 2.0)']
    """
    return sum((upwind(SR(expression), SR(axis), SR(item)) for axis, item in enumerate(velocity)), SR(0))


def inverse_helmholtz(expression: object, screening: object = 0, boundary: object = None) -> Expression:
    """Return the solution `psi` of a modified-Helmholtz equation.

    The equation is `(laplacian - screening**2) psi = expression`. Without screening it is the Poisson equation
    `laplacian(psi) = expression`, the inverse Laplacian.

    Args:
        expression:
            An expression of the states and parameters of a field.

        screening:
            The screening, a non-negative expression of the parameters, or 0 for the Poisson equation.

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
        >>> from skcomplex.core.components.laws.utils import inverse_helmholtz
        >>> inverse_helmholtz(var('q'), 2)
        inverse_helmholtz(q, 2)
        >>> inverse_helmholtz(var('q'))
        inverse_helmholtz(q, 0)
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


def collision_integral(density: object, kernel: Callable[[Expression, Expression], object]) -> Expression:
    """Return the binary collision integral of a density with a collision kernel.

    The integral is `Q(f, f)(v) = int int B(|g|, cos t) (f(v') f(v'*) - f(v) f(v*)) ds dv*` over the two velocity
    axes of the phase space of the density, with the relative speed `|g|` and the cosine `cos t` of the deflection
    angle. The kernel is called once with two real symbols for the speed and the cosine, and the call keeps its
    expression, so a rate law that reads the integral needs no other argument. The kernel returns an expression of the
    speed, the cosine, and numbers, written with Sage functions rather than NumPy functions.

    Args:
        density:
            The symbol of a density on a phase space with two interval velocity axes.

        kernel:
            The function `B(speed, cosine)` of the collision kernel, such as `lambda speed, cosine: 1 / (2 * pi)` for
            Maxwell molecules.

    Returns:
        integral:
            The unevaluated call of the integral, which the compiler lowers to the fast spectral method.

    Examples:
        >>> from sage.all__sagemath_symbolics import pi, var
        >>> from skcomplex.core.components.laws.utils import collision_integral
        >>> collision_integral(var('f'), lambda speed, cosine: 1 / (2 * pi))
        collision_integral(f, 1/2/pi)
    """
    return COLLISION_INTEGRAL(SR(density), SR(kernel(COLLISION_SPEED, COLLISION_COSINE)))


def displacement(axis: int) -> Expression:
    """Return the symbol of the displacement along an axis.

    It is the variable a convolution kernel is written in.

    Args:
        axis:
            The axis of the lattice.

    Returns:
        displacement:
            The real symbol `displacement_<axis>`.

    Examples:
        >>> from skcomplex.core.components.laws.utils import displacement
        >>> displacement(0) ** 2
        displacement_0^2
    """
    return SR.var(f'displacement_{axis}', domain='real')


def convolve(expression: object, kernel: object, radius: object = None) -> Expression:
    """Return the convolution of a field expression with a kernel of the displacement.

    It is the sum over sites times volume.

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
        >>> from skcomplex.core.components.laws.utils import convolve, displacement
        >>> convolve(var('u'), exp(-displacement(0) ** 2), 3)
        convolve(u, e^(-displacement_0^2), 3)
    """
    return CONVOLVE(SR(expression), SR(kernel), oo if radius is None else SR(radius))


def shifted(expression: object, offset: Sequence[object], order: int = 1) -> Expression:
    """Return a field expression read at a physical displacement from every site.

    The displacement has one component per axis.

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
        >>> from skcomplex.core.components.laws.utils import shifted
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
