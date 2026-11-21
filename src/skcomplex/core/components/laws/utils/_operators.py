"""Define the differential operators of fields, such as the gradient."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copyreg
from collections.abc import Callable, Sequence

from sage.all__sagemath_symbolics import SR, Expression, oo, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.symbolic.function import BuiltinFunction

from ....enums import BoundaryKind
from ....errors import ValidationError

COLLISION_SPEED = SR.symbol('collision_speed', domain='real')
COLLISION_COSINE = SR.symbol('collision_cosine', domain='real')
INVERSION_BOUNDARIES = (BoundaryKind.PERIODIC, BoundaryKind.ZERO_FLUX, BoundaryKind.FIXED)
ORDERS = (1, 3)


class _OperatorFunction(BuiltinFunction):
    """Operator kept unevaluated in expressions until the compiler lowers it.

    The name, the number of operands, 0 for any number, and the LaTeX name are those of Sage's `BuiltinFunction`. Sage
    pickles a `BuiltinFunction` by calling its class with no argument, so every operator class registers with `copyreg`
    a reduction that rebuilds it from the arguments it was built with, which lets expressions that hold it reach other
    processes.
    """

    @staticmethod
    def _reduce(operator: '_OperatorFunction') -> tuple[type['_OperatorFunction'], tuple[str, int, str | None]]:
        """Return the class and the arguments that rebuild an operator."""
        return type(operator), operator._arguments

    def __init__(self, name: str, nargs: int = 0, latex_name: str | None = None) -> None:
        super().__init__(name, nargs=nargs, latex_name=latex_name)
        self._arguments = (name, nargs, latex_name)
        copyreg.pickle(type(self), _OperatorFunction._reduce)


laplacian = _OperatorFunction('laplacian', nargs=1, latex_name=r'\nabla^2')
partial = _OperatorFunction('partial', nargs=2)
upwind = _OperatorFunction('upwind', nargs=3)
COLLISION_INTEGRAL = _OperatorFunction('collision_integral', nargs=2)
INVERSE_HELMHOLTZ = _OperatorFunction('inverse_helmholtz')
CONVOLVE = _OperatorFunction('convolve', nargs=3)
SHIFTED = _OperatorFunction('shifted')


def gradient(expression: object, n_dim: int = 2) -> FreeModuleElement:
    """Return the gradient of a field, its derivative along every axis of a lattice.

    It gives one `partial(expression, axis)` per axis, such as `(partial(u, 0), partial(u, 1))` for a field `u` in a
    plane, which the compiler turns into differences between neighbouring sites. It gives the slope of a landscape, the
    pull of a chemical on cells that follow it, and the force of a pressure on a fluid.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states and parameters of a field.

        n_dim:
            The number of dimensions of the lattice.

    Returns:
        gradient:
            One derivative per dimension.

    See Also:
        divergence: Sums the derivatives of a vector field.
        laplacian: Gives the divergence of the gradient.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import gradient
        >>> gradient(var('u'))
        (partial(u, 0), partial(u, 1))
    """
    return vector(SR, [partial(SR(expression), SR(axis)) for axis in range(n_dim)])


def divergence(components: Sequence[object] | FreeModuleElement) -> Expression:
    """Return the divergence of a vector field, the net outflow at every site.

    It sums the derivative of entry `i` along axis `i`, such as `partial(u, 0) + partial(2*u, 1)` for `[u, 2 * u]`. The
    rate of a density is minus the divergence of its flux, so what leaves a site arrives at its neighbours. It moves
    heat through a plate, nutrients through a tissue, and traffic along a road.

    Read more in the user guide, under Laws.

    Args:
        components:
            One expression per dimension, such as a flux.

    Returns:
        divergence:
            The divergence.

    See Also:
        gradient: Gives the derivative along every axis.
        advection: Moves a field with a velocity.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import divergence
        >>> u = var('u')
        >>> sorted(str(term) for term in divergence([u, 2 * u]).operands())
        ['partial(2*u, 1)', 'partial(u, 0)']
    """
    return sum((partial(SR(item), SR(axis)) for axis, item in enumerate(components)), SR(0))


def advection(expression: object, velocity: Sequence[object] | FreeModuleElement) -> Expression:
    """Return the rate at which a velocity carries a field along, for one site.

    It sums `upwind(u, axis, v)` over the axes, the velocity times the slope of the field taken on the side the flow
    comes from, which keeps a moving front from rippling, such as `upwind(u, 0, 1.0) + upwind(u, 1, 2.0)` for a flow
    `(1, 2)`. A rate law subtracts it from the rate of the field. It carries a pollutant down a river, heat with the
    wind, and cars along a road.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states and parameters of a field.

        velocity:
            One entry of the velocity per dimension, numbers or expressions.

    Returns:
        advection:
            The advection term, to subtract from the rate of the field.

    See Also:
        divergence: Gives the net outflow of a flux.
        shifted: Reads a field at a distance from every site.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import advection
        >>> sorted(str(term) for term in advection(var('u'), [1.0, 2.0]).operands())
        ['upwind(u, 0, 1.0)', 'upwind(u, 1, 2.0)']
    """
    return sum((upwind(SR(expression), SR(axis), SR(item)) for axis, item in enumerate(velocity)), SR(0))


def inverse_helmholtz(expression: object, screening: object = 0, boundary: object = None) -> Expression:
    """Return the field `psi` with `(laplacian - screening**2) psi = expression`.

    The compiler solves the equation over the whole lattice at every step, so a law reads a potential that its sources
    set at once, such as `inverse_helmholtz(q, 2)` for the potential of the charge `q` screened over a distance of one
    half. Without screening it solves the Poisson equation `laplacian(psi) = expression`. It gives the electric
    potential of charges, the gravity of a mass, and the signal that cells emit and that decays as it spreads.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states and parameters of a field.

        screening:
            The screening, an expression of the parameters of 0 or more, or 0 for the Poisson equation.

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

    See Also:
        laplacian: Gives the operator it inverts.
        leray_projection: Removes the outflow of a vector field with it.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import inverse_helmholtz
        >>> inverse_helmholtz(var('q'), 2)
        inverse_helmholtz(q, 2)
    """
    if boundary is None:
        return INVERSE_HELMHOLTZ(SR(expression), SR(screening))
    kind, *values = boundary if isinstance(boundary, tuple) else (boundary,)
    if kind not in INVERSION_BOUNDARIES or (kind == 'fixed') != bool(values) or len(values) > 1:
        error_msg = (
            f"The boundary of inverse_helmholtz must be 'periodic', 'zero-flux', ('fixed', value), or None, but it is "
            f'{boundary!r}.'
        )
        raise ValidationError(error_msg)
    code = SR(INVERSION_BOUNDARIES.index(kind))
    return INVERSE_HELMHOLTZ(SR(expression), SR(screening), code, SR(values[0] if values else 0))


def collision_integral(density: object, kernel: Callable[[Expression, Expression], object]) -> Expression:
    """Return the rate at which collisions between pairs change a density of speeds.

    It is the Boltzmann integral `Q(f, f)(v) = ∫∫ B(|g|, cos t) (f(v') f(v'*) - f(v) f(v*)) ds dv*` over the two
    velocity axes of the phase space of the density, with the relative speed `|g|` and the cosine `cos t` of the angle
    the collision turns by. The kernel `B` is called once with two real symbols for the speed and the cosine and must
    use Sage functions, such as `lambda speed, cosine: 1 / (2 * pi)` for Maxwell molecules, and the call keeps its
    expression. It relaxes a gas toward equilibrium, mixes the speeds of granular matter, and spreads the opinions of
    agents who meet in pairs.

    Read more in the user guide, under Laws.

    Args:
        density:
            The symbol of a density on a phase space with two interval velocity axes.

        kernel:
            The function `B(speed, cosine)` of the collision kernel, such as `lambda speed, cosine: 1 / (2 * pi)` for
            Maxwell molecules.

    Returns:
        integral:
            The unevaluated call of the integral, which the compiler computes with the fast spectral method.

    See Also:
        bilinear: Gives the rate of meetings between classes of two densities.

    Examples:
        >>> from sage.all__sagemath_symbolics import pi, var
        >>> from skcomplex.core.components.laws.utils import collision_integral
        >>> collision_integral(var('f'), lambda speed, cosine: 1 / (2 * pi))
        collision_integral(f, 1/2/pi)
    """
    return COLLISION_INTEGRAL(SR(density), SR(kernel(COLLISION_SPEED, COLLISION_COSINE)))


def displacement(axis: int) -> Expression:
    """Return the symbol of the distance along an axis, for a convolution kernel.

    A kernel of `convolve` is written in these symbols, such as `exp(-displacement(0) ** 2)` for a bell curve of the
    distance along axis 0. It shapes how far seeds disperse, how far neurons reach, and how a blur spreads a pixel.

    Read more in the user guide, under Laws.

    Args:
        axis:
            The axis of the lattice.

    Returns:
        displacement:
            The real symbol `displacement_<axis>`.

    See Also:
        convolve: Sums a field against a kernel of these symbols.

    Examples:
        >>> from skcomplex.core.components.laws.utils import displacement
        >>> displacement(0) ** 2
        displacement_0^2
    """
    return SR.var(f'displacement_{axis}', domain='real')


def convolve(expression: object, kernel: object, radius: object = None) -> Expression:
    """Return the sum of a field around every site, weighted by a kernel of distance.

    At every site it sums the field at every other site times the kernel at their distance and the volume of a cell,
    such as `convolve(u, exp(-displacement(0) ** 2), 3)` for a bell curve that ends at distance 3. It gives the seeds
    that land around a plant, the input a neuron gets from its neighbours, and the blur of an image.

    Read more in the user guide, under Laws.

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

    See Also:
        displacement: Gives the symbols the kernel is written in.
        integral_transform: Integrates against a kernel of two points.

    Examples:
        >>> from sage.all__sagemath_symbolics import var, exp
        >>> from skcomplex.core.components.laws.utils import convolve, displacement
        >>> convolve(var('u'), exp(-displacement(0) ** 2), 3)
        convolve(u, e^(-displacement_0^2), 3)
    """
    return CONVOLVE(SR(expression), SR(kernel), oo if radius is None else SR(radius))


def shifted(expression: object, offset: Sequence[object], order: int = 1) -> Expression:
    """Return a field read at a given distance from every site.

    It reads the field at the point `offset` away from every site, with one distance per axis, and fills in between
    sites with a straight line or a cubic curve, such as `shifted(u, [v, 1])` for the field at `v` along the first axis
    and 1 along the second. It moves a field back along a flow, reads the past position of a moving front, and shifts a
    population along an axis of age.

    Read more in the user guide, under Laws.

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

    See Also:
        advection: Carries a field along with a velocity.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import shifted
        >>> shifted(var('u'), [var('v'), 1])
        shifted(u, v, 1)
    """
    if order not in ORDERS:
        error_msg = f'The order of a shifted read is {order!r}. Use 1 for linear or 3 for cubic interpolation.'
        raise ValidationError(error_msg)
    orders = [SR(order)] if order != 1 else []
    return SHIFTED(SR(expression), *[SR(item) for item in offset], *orders)
