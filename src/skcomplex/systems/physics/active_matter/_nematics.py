"""Active liquid crystals.

The module covers a dry active nematic and active nematic and polar gels.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, identity_matrix, oo, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet
from sage.structure.element import Matrix

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import ConservationInteraction, GradientInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import (
    advection,
    antisymmetric_part,
    gradient,
    inverse_helmholtz,
    laplacian,
    leray_projection,
    symmetric_part,
    tensor_divergence,
    vector_gradient,
)
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, RunSettings, System
from ....core.engine.backends import NumbaBackend
from ....core.engine.solvers import RK4Solver
from ....core.errors import TensorShapeError, ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class NematicDensitySiteElement(Element):
    """Site of a dry active nematic.

    Its alignment tensor grows with the traceless curvature of the density.

    The rate of the tensor from the density is `beta (partial_i partial_j rho - delta_ij laplacian(rho) / 2)`. The
    Landau-de Gennes energy and the current of the density are interactions of the system.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        rho:
            The initial density.

        Q:
            The initial alignment tensor, its independent components `Q_xx` and `Q_xy`.

        beta:
            The coupling of the tensor to the curvature of the density.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import NematicDensitySiteElement
        >>> NematicDensitySiteElement('site').beta
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[npt.ArrayLike, StateSpec()] = 1.0,
        Q: Annotated[npt.ArrayLike, StateSpec(tensor='traceless')] = 0.0,
        beta: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.Q = Q
        self.beta = beta

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the tensor from the curvature of the density.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of `Q`.
        """
        s = namespace
        curvature = symmetric_part(vector_gradient(gradient(s.rho)))
        traceless = curvature - curvature.trace() / 2 * identity_matrix(SR, 2)
        return [RateLaw(s.Q, s.beta * traceless)]


class PolarGelSiteElement(Element):
    """Site of an active polar gel.

    A polarisation `p` is carried by an overdamped Stokes flow.

    The polarisation follows `p_t = -corotational(p, u, xi) + h / gamma`, with the molecular field
    `h = -a p - c |p|^2 p + K laplacian(p)`. The velocity solves `eta laplacian(u) - grad(P) + div(sigma) = 0`,
    `div(u) = 0`, with the stress `sigma = -xi (p h + h p) / 2 + (p h - h p) / 2 - zeta p p`, as in Kruse et al.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        p:
            The initial polarisation.

        a:
            The coefficient of `|p|^2 / 2` of the energy, negative in the polar phase.

        c:
            The coefficient of `|p|^4 / 4` of the energy.

        K:
            The elastic constant.

        xi:
            The flow-alignment parameter.

        gamma:
            The rotational viscosity.

        eta:
            The viscosity of the flow.

        zeta:
            The activity.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import PolarGelSiteElement
        >>> PolarGelSiteElement('site').xi
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        a: Annotated[float, ParameterSpec()] = -1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        xi: Annotated[float, ParameterSpec()] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        zeta: Annotated[float, ParameterSpec()] = -0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p
        self.a = a
        self.c = c
        self.K = K
        self.xi = xi
        self.gamma = gamma
        self.eta = eta
        self.zeta = zeta

    @staticmethod
    def _molecular_field(s: SimpleNamespace) -> FreeModuleElement:
        """Return the molecular field of the polarisation at a site."""
        return -s.a * s.p - s.c * s.p.dot_product(s.p) * s.p + s.K * s.p.apply_map(laplacian)

    @staticmethod
    def _velocity(s: SimpleNamespace) -> FreeModuleElement:
        """Return the velocity of the Stokes flow of the stresses at a site.

        The stresses are the elastic and active ones.
        """
        field = PolarGelSiteElement._molecular_field(s)
        outer = s.p.outer_product(field)
        stress = -s.xi * (outer + outer.transpose()) / 2 + (outer - outer.transpose()) / 2
        stress -= s.zeta * s.p.outer_product(s.p)
        force = tensor_divergence(stress)
        return leray_projection(force.apply_map(lambda item: inverse_helmholtz(-item / s.eta)))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the polarisation.

        The rate is its transport in the flow and its relaxation.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of `p`.
        """
        s = namespace
        return [RateLaw(s.p, -corotational(s.p, self._velocity(s), s.xi) + self._molecular_field(s) / s.gamma)]


class NematicGelSiteElement(Element):
    """Site of an active nematic gel.

    A symmetric traceless tensor `Q` is carried by an overdamped Stokes flow.

    The tensor follows `Q_t = -corotational(Q, u, xi) + H / gamma`, with the molecular field `H` of the Landau-de Gennes
    energy. The velocity solves `eta laplacian(u) - grad(P) + div(sigma) = 0`, `div(u) = 0`, with the stress
    `sigma = -xi H + Q H - H Q - zeta Q`. An activity `zeta > 0` is extensile and an activity `zeta < 0` contractile.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        Q:
            The initial tensor, its independent components `Q_xx` and `Q_xy`.

        a:
            The coefficient of `tr(Q^2) / 2` of the energy, negative in the ordered phase.

        c:
            The coefficient of `tr(Q^2)^2 / 4` of the energy.

        K:
            The elastic constant.

        xi:
            The flow-alignment parameter.

        gamma:
            The rotational viscosity.

        eta:
            The viscosity of the flow.

        zeta:
            The activity.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import NematicGelSiteElement
        >>> NematicGelSiteElement('site').zeta
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Q: Annotated[npt.ArrayLike, StateSpec(tensor='traceless')] = 0.0,
        a: Annotated[float, ParameterSpec()] = -1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        xi: Annotated[float, ParameterSpec()] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        zeta: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Q = Q
        self.a = a
        self.c = c
        self.K = K
        self.xi = xi
        self.gamma = gamma
        self.eta = eta
        self.zeta = zeta

    @staticmethod
    def _velocity(s: SimpleNamespace) -> FreeModuleElement:
        """Return the velocity of the Stokes flow of the stresses at a site.

        The stresses are the elastic and active ones.
        """
        field = LandauDeGennesEnergyInteraction._molecular_field(s.Q, s.a, s.c, s.K)
        stress = -s.xi * field + s.Q * field - field * s.Q - s.zeta * s.Q
        force = tensor_divergence(stress)
        return leray_projection(force.apply_map(lambda item: inverse_helmholtz(-item / s.eta)))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the tensor, its transport in the flow and its relaxation.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of `Q`.
        """
        s = namespace
        field = LandauDeGennesEnergyInteraction._molecular_field(s.Q, s.a, s.c, s.K)
        return [RateLaw(s.Q, -corotational(s.Q, self._velocity(s), s.xi) + field / s.gamma)]


class LandauDeGennesEnergyInteraction(GradientInteraction):
    """Landau-de Gennes free energy gradient flow of a symmetric traceless tensor `Q`.

    The free energy is `F = integral a tr(Q^2) / 2 + c tr(Q^2)^2 / 4 + K |grad Q|^2 / 2`. Its negative functional
    derivative among symmetric traceless tensors is the molecular field `H = -a Q - c tr(Q^2) Q + K laplacian(Q)`, and
    the tensor relaxes as `Q_t = H / gamma`. A uniform tensor `S (n n - I / 2)` in two dimensions with `a < 0` relaxes
    to the scalar order `S = sqrt(-2 a / c)`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field.

        state:
            The name of the state of the field that holds the tensor.

        a:
            The coefficient of `tr(Q^2) / 2`, negative in the ordered phase.

        c:
            The coefficient of `tr(Q^2)^2 / 4`.

        K:
            The elastic constant.

        gamma:
            The rotational viscosity, which divides the molecular field.

        quadratic:
            A function of the namespace that returns the coefficient of `tr(Q^2) / 2` instead of `a`, such as an
            expression of a density, or `None` for `a`.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, matrix, var
        >>> from skcomplex.systems.physics.active_matter import LandauDeGennesEnergyInteraction
        >>> q, a, c = var('q, a, c')
        >>> tensor = matrix(SR, [[q, 0], [0, -q]])
        >>> LandauDeGennesEnergyInteraction._molecular_field(tensor, a, c, 0)[0, 0].factor()
        -(2*c*q^2 + a)*q
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'Q',
        a: Annotated[float, ParameterSpec()] = -1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        quadratic: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label, target=target, states=state, variational=True)
        self.state = state
        self.a = a
        self.c = c
        self.K = K
        self.gamma = gamma
        self.quadratic = quadratic

    @staticmethod
    def _density(tensor: Matrix, a: object, c: object, K: object) -> Expression:
        """Return the density of the free energy, with the gradient term by parts."""
        square = (tensor * tensor).trace()
        return a * square / 2 + c * square**2 / 4 - K * (tensor * tensor.apply_map(laplacian)).trace() / 2

    @classmethod
    def _molecular_field(cls, tensor: Matrix, a: object, c: object, K: object) -> Matrix:
        """Return the molecular field of a tensor.

        The field is the negative functional derivative of the free energy.
        """
        energy = cls('energy', target='field')
        return -energy._get_matrix_derivative(cls._density(tensor, a, c, K), tensor)

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the density of the free energy over the rotational viscosity.

        Its gradient is `H / gamma`.

        Args:
            namespace:
                The namespace of the symbols the density can use.

        Returns:
            density:
                The density that the tensor moves down.
        """
        quadratic = namespace.a if self.quadratic is None else self.quadratic(namespace)
        tensor = getattr(namespace.target, self.state)
        return self._density(tensor, quadratic, namespace.c, namespace.K) / namespace.gamma


GEL_ORDERS = ('nematic', 'polar')
NEMATIC_BOX = 10.0
NEMATIC_CRITICAL_DENSITY = 3 * np.pi / 2
NEMATIC_DENSITY = 2 * NEMATIC_CRITICAL_DENSITY
NEMATIC_ACTIVITY = 3.0
NEMATIC_XI = 0.1


def _check_matrix(utility: str, value: object) -> Matrix:
    """Return a square matrix, or raise an error that names the utility and the value."""
    if not isinstance(value, Matrix) or not value.is_square():
        error_msg = f'{utility} needs a square matrix, but got {value}, which has another shape.'
        raise TensorShapeError(error_msg)
    return value


def _check_vector(utility: str, value: object) -> FreeModuleElement:
    """Return a vector, or a sequence as a vector.

    Any other value raises an error that names the utility and the value.
    """
    if isinstance(value, list | tuple):
        return vector(SR, [SR(item) for item in value])
    if not isinstance(value, FreeModuleElement):
        error_msg = f'{utility} needs a vector, but got {value}, which has another shape.'
        raise TensorShapeError(error_msg)
    return value


def _advected(expression: Expression, velocity: FreeModuleElement) -> Expression:
    """Return the advection of an entry of a matrix.

    The advection is 0 for a number, whose derivatives are 0.
    """
    return SR(0) if SR(expression).is_numeric() else advection(expression, velocity)


def corotational(
    value: Matrix | FreeModuleElement,
    velocity: Sequence[object] | FreeModuleElement,
    xi: object = 0,
) -> Matrix | FreeModuleElement:
    """Return the transport of a symmetric tensor or a vector by a flow.

    The flow rotates and aligns it.

    A symmetric tensor `T` gets the Beris-Edwards transport `u.grad(T) - S`, where
    `S = (xi E + W)(T + I/d) + (T + I/d)(xi E - W) - 2 xi (T + I/d) tr(T grad(u))`, with the strain rate `E`, the
    vorticity tensor `W`, the antisymmetric part of `vector_gradient(u)`, and the dimension `d`. A vector `p` gets
    `u.grad(p) - W p - xi E p`, so both turn with the flow.

    Args:
        value:
            A symmetric square matrix, or a vector.

        velocity:
            One velocity component per dimension.

        xi:
            The flow-alignment parameter.

    Returns:
        transport:
            The matrix or the vector to subtract from the rate of the value.

    Raises:
        TensorShapeError:
            The value is neither a square matrix nor a vector.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, matrix, var
        >>> from skcomplex.systems.physics.active_matter._nematics import corotational
        >>> a, b = var('a, b')
        >>> corotational(matrix(SR, [[a, b], [b, -a]]), [var('u'), 0]).dimensions()
        (2, 2)
    """
    flow = _check_vector('corotational', velocity)
    size = len(flow)
    gradients = vector_gradient(flow)
    strain, vorticity = symmetric_part(gradients), antisymmetric_part(gradients)
    if isinstance(value, FreeModuleElement):
        advected = vector(SR, [_advected(item, flow) for item in value])
        return advected - vorticity * value - SR(xi) * (strain * value)
    tensor = _check_matrix('corotational', value)
    advected_tensor = tensor.apply_map(lambda item: _advected(item, flow))
    shifted_tensor = tensor + identity_matrix(SR, size) / size
    aligned = (SR(xi) * strain + vorticity) * shifted_tensor + shifted_tensor * (SR(xi) * strain - vorticity)
    stretching = 2 * SR(xi) * (tensor * gradients).trace() * shifted_tensor
    return advected_tensor - aligned + stretching


def upper_convected(tensor: Matrix, velocity: Sequence[object] | FreeModuleElement) -> Matrix:
    """Return the upper-convected transport of a rank-2 tensor.

    The transport is `u.grad(T) - grad(u)^T T - T grad(u)`.

    Args:
        tensor:
            A square matrix of expressions.

        velocity:
            One velocity component per dimension.

    Returns:
        transport:
            The matrix to subtract from the rate of the tensor.

    Raises:
        TensorShapeError:
            The tensor is not a square matrix or the velocity is not a vector.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, identity_matrix, var
        >>> from skcomplex.systems.physics.active_matter._nematics import upper_convected
        >>> upper_convected(identity_matrix(SR, 2), [var('u'), var('v')])[0, 0]
        -2*partial(u, 0)
    """
    checked = _check_matrix('upper_convected', tensor)
    flow = _check_vector('upper_convected', velocity)
    gradients = vector_gradient(flow)
    advected = checked.apply_map(lambda item: _advected(item, flow))
    return advected - gradients.transpose() * checked - checked * gradients


def _gel_velocity(s: SimpleNamespace) -> FreeModuleElement:
    """Return the velocity of the flow of a gel at a site.

    The velocity depends on the order the namespace holds.
    """
    return PolarGelSiteElement._velocity(s) if hasattr(s, 'p') else NematicGelSiteElement._velocity(s)


def _velocity_x(s: SimpleNamespace) -> Expression:
    """Return the velocity along the first axis at a site."""
    return _gel_velocity(s)[0]


def _velocity_y(s: SimpleNamespace) -> Expression:
    """Return the velocity along the second axis at a site."""
    return _gel_velocity(s)[1]


def _sites(values: np.ndarray) -> np.ndarray:
    """Return the values of every site, the observed field itself."""
    return np.array(values)


class DryActiveNematicsSystem(BaseSystem):
    """A dry active nematic.

    A conserved density is coupled to a symmetric traceless alignment tensor.

    On a periodic lattice the density and the tensor follow Section II.2 of Marchetti et al.,

        rho_t = -div(J),    J = -D grad(rho) - current div(Q),
        Q_t = H / gamma + beta (partial_i partial_j rho - delta_ij laplacian(rho) / 2),

    with the molecular field `H` of the Landau-de Gennes energy, whose coefficient `a (1 - rho / rho_c)` turns negative
    above the critical density `rho_c`. The current along `div(Q)` is the active current, which no free energy gives.
    The system has the fields `nematic.rho` and `nematic.Q`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        density:
            The mean density, which the dynamics conserve.

        critical_density:
            The density at which the isotropic state loses stability.

        a:
            The scale of the coefficient of `tr(Q^2) / 2` of the energy.

        c:
            The coefficient of `tr(Q^2)^2 / 4` of the energy.

        K:
            The elastic constant.

        gamma:
            The rotational viscosity.

        D:
            The diffusion coefficient of the density.

        current:
            The coefficient of the active current along `div(Q)`.

        beta:
            The coupling of the tensor to the curvature of the density.

        perturbation:
            The standard deviation of the initial noise of the density and the tensor.

        random_state:
            The seed or generator of the initial noise.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import DryActiveNematicsSystem
        >>> DryActiveNematicsSystem((8, 8)).prepare().system_.elements[0].values['beta']
        0.5
    """

    field = 'physics/active_matter'
    topics = ('field', 'active matter', 'nematic', 'tensor', 'liquid crystal', 'giant number fluctuations', 'pde')
    references = ('Marchetti et al. 2013', 'arXiv:1207.2929')
    until = 100.0
    dt = 5.0
    solver = RK4Solver(dt=0.02)
    backend = NumbaBackend()

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        density: float = 1.5,
        critical_density: float = 1.0,
        a: float = 1.0,
        c: float = 2.0,
        K: float = 1.0,
        gamma: float = 1.0,
        D: float = 0.5,
        current: float = 0.5,
        beta: float = 0.5,
        perturbation: float = 0.01,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.density = density
        self.critical_density = critical_density
        self.a = a
        self.c = c
        self.K = K
        self.gamma = gamma
        self.D = D
        self.current = current
        self.beta = beta
        self.perturbation = perturbation
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return a dry active nematic.

        A conserved density is coupled to a symmetric traceless alignment tensor.
        """
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        sites = int(np.prod(shape))
        values = {
            'rho': self.density + self.perturbation * rng.standard_normal(sites),
            'Q': self.perturbation * rng.standard_normal((sites, 2)),
            'beta': self.beta,
        }
        nematic = FieldElement('nematic', element=NematicDensitySiteElement, space=LatticeSpace(shape), values=values)
        energy = LandauDeGennesEnergyInteraction(
            'energy',
            target='nematic',
            a=self.a,
            c=self.c,
            K=self.K,
            gamma=self.gamma,
            quadratic=lambda s: s.a * (1 - s.target.rho / self.critical_density),
        )

        def flux(s: SimpleNamespace) -> list[Expression]:
            """Return the current of the density, its diffusion and its active current."""
            active = tensor_divergence(s.target.Q)
            return [
                -self.D * item - self.current * other
                for item, other in zip(gradient(s.target.rho), active, strict=True)
            ]

        transport = ConservationInteraction(
            'transport',
            target='nematic',
            state='rho',
            flux=flux,
            limiter=None,
            riemann=None,
        )
        return System(self.label, elements=[nematic], interactions=[energy, transport])


class ActiveGelsSystem(BaseSystem):
    """An active gel.

    A nematic tensor or a polarisation is transported in the Stokes flow of its own stresses.

    On a periodic lattice with spectral operators, the order parameter follows its co-rotational derivative in the flow,
    Section III of Marchetti et al. The flow solves the overdamped Stokes equation of the elastic stress and the active
    stress, `-zeta Q` for the nematic and `-zeta p p` for the polar gel, and the elastic energy relaxes the order. An
    extensile nematic, `zeta > 0`, is unstable to bend, and a contractile polar gel, `zeta < 0`, is unstable to splay
    above a threshold.

    The defaults of the nematic are the chaotic state of Giomi et al. 2012, cited by Marchetti et al., whose free energy
    has the same form: a box of side 10, the critical density `3 pi / 2`, the density twice that, which give `a` and
    `c`, the activity `3` times the squared density, and the flow alignment 0.1, in units of the elastic constant, the
    rotational viscosity, and the viscosity. The density stays uniform here. At these defaults pairs of `+1/2` and
    `-1/2` defects move in a chaotic flow, active turbulence.

    The nematic starts from the tensor along the first axis at the order `sqrt(-2 a / c)`, and the polar gel from the
    polarisation along the first axis at the length `sqrt(-a / c)`, both with noise. With `film=True` the polarisation
    along the second axis is odd about the sites 0 and `shape[1] / 2` of the second axis, and the other one even, which
    holds the polarisation along the first axis at these two walls, with stress-free flow there. The system has the
    field `gel.Q` or `gel.p`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice, in two dimensions, or `None` for `(128, 128)`
            for the nematic and `(32, 32)` for the polar gel.

        order:
            `'nematic'` or `'polar'`.

        spacing:
            The distance between neighbouring sites, or `None` for a nematic box of side 10 along the first axis and 1
            for the polar gel.

        a:
            The coefficient of the quadratic energy, negative in the ordered phase, or `None` for `-3 pi / 4` for the
            nematic and -1 for the polar gel.

        c:
            The coefficient of the quartic energy, or `None` for `3 pi` for the nematic and 1 for the polar gel.

        K:
            The elastic constant.

        xi:
            The flow-alignment parameter, or `None` for 0.1 for the nematic and 0 for the polar gel.

        gamma:
            The rotational viscosity.

        eta:
            The viscosity of the flow.

        zeta:
            The activity, or `None` for `3 (3 pi)^2` for the nematic and -0.5 for the polar gel.

        film:
            Whether the polar gel lies between walls of fixed polarisation.

        perturbation:
            The standard deviation of the initial noise of the order parameter, or `None` for 0.1 for the nematic and
            0.01 for the polar gel.

        random_state:
            The seed or generator of the initial noise.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The order is unknown, or a nematic is asked to lie in a film.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveGelsSystem
        >>> ActiveGelsSystem((8, 8), order='polar').prepare().system_.elements[0].values['zeta']
        -0.5
        >>> ActiveGelsSystem(order='polar').prepare().system_.elements[0].space.shape
        (32, 32)
    """

    field = 'physics/active_matter'
    topics = ('field', 'active matter', 'nematic', 'polar', 'tensor', 'gel', 'stokes flow', 'active turbulence', 'pde')
    references = (
        'Marchetti et al. 2013',
        'arXiv:1207.2929',
        'Kruse et al. 2005',
        'Voituriez, Joanny, and Prost 2005',
        'Giomi et al. 2012',
        'arXiv:1110.4338',
    )
    until = 0.5
    dt = 0.025
    solver = RK4Solver(dt=5e-4)
    observables = (
        ObservableSpec('u_x', _velocity_x, group='gel', reduce=_sites),
        ObservableSpec('u_y', _velocity_y, group='gel', reduce=_sites),
    )

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        *,
        order: str = 'nematic',
        spacing: float | None = None,
        a: float | None = None,
        c: float | None = None,
        K: float = 1.0,
        xi: float | None = None,
        gamma: float = 1.0,
        eta: float = 1.0,
        zeta: float | None = None,
        film: bool = False,
        perturbation: float | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.order = order
        self.spacing = spacing
        self.a = a
        self.c = c
        self.K = K
        self.xi = xi
        self.gamma = gamma
        self.eta = eta
        self.zeta = zeta
        self.film = film
        self.perturbation = perturbation
        self.random_state = random_state

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        They are longer and coarser for the polar gel, whose flow is slow.
        """
        settings = super().get_run_settings()
        if self.order == 'polar':
            settings['until'] = 200.0
            settings['dt'] = 1.0
            settings['solver'] = RK4Solver(dt=0.05)
        return settings

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        if self.order not in GEL_ORDERS:
            error_msg = f'The order of the gel is {self.order!r}. Use one of {", ".join(GEL_ORDERS)}.'
            raise ValidationError(error_msg)
        if self.film and self.order != 'polar':
            error_msg = 'A film of the gel holds a polarisation at its walls. Use order=polar.'
            raise ValidationError(error_msg)
        polar = self.order == 'polar'
        shape = ((32, 32) if polar else (128, 128)) if self.shape is None else tuple(self.shape)
        sites = int(np.prod(shape))
        rng = np.random.default_rng(self.random_state)
        quadratic = (-1.0 if polar else (NEMATIC_CRITICAL_DENSITY - NEMATIC_DENSITY) / 2) if self.a is None else self.a
        cubic = (1.0 if polar else NEMATIC_DENSITY) if self.c is None else self.c
        activity = (-0.5 if polar else NEMATIC_ACTIVITY * NEMATIC_DENSITY**2) if self.zeta is None else self.zeta
        alignment = (0.0 if polar else NEMATIC_XI) if self.xi is None else self.xi
        distance = (1.0 if polar else NEMATIC_BOX / shape[0]) if self.spacing is None else self.spacing
        amplitude = (0.01 if polar else 0.1) if self.perturbation is None else self.perturbation
        noise = amplitude * rng.standard_normal((sites, 2))
        size = np.sqrt(-quadratic / cubic) if polar else np.sqrt(-2 * quadratic / cubic) / 2
        start = np.column_stack([np.full(sites, size), np.zeros(sites)]) + noise if quadratic < 0 else noise
        values = {
            'p' if polar else 'Q': start,
            'a': quadratic,
            'c': cubic,
            'K': self.K,
            'xi': alignment,
            'gamma': self.gamma,
            'eta': self.eta,
            'zeta': activity,
        }
        symmetry = {'p': ((None, 'even'), (None, 'odd'))} if self.film else None
        gel = FieldElement(
            'gel',
            element=PolarGelSiteElement if polar else NematicGelSiteElement,
            space=LatticeSpace(shape, spacing=distance),
            values=values,
            discretization='spectral',
            symmetry=symmetry,
        )
        return System(self.label, elements=[gel])
