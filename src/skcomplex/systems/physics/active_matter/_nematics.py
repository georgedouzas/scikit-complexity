"""Active liquid crystals: a dry active nematic and active nematic and polar gels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.components.elements import FieldElement
from ....core.components.interactions import ConservationInteraction
from ....core.components.laws.utils import gradient, tensor_divergence
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.backends import NumbaBackend
from ....core.engine.solvers import RK4Solver
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._landau_de_gennes_energy import LandauDeGennesEnergyInteraction
from ._nematic_density_site import NematicDensitySiteElement
from ._nematic_gel_site import NematicGelSiteElement
from ._polar_gel_site import PolarGelSiteElement

GEL_ORDERS = ('nematic', 'polar')
NEMATIC_BOX = 10.0
NEMATIC_CRITICAL_DENSITY = 3 * np.pi / 2
NEMATIC_DENSITY = 2 * NEMATIC_CRITICAL_DENSITY
NEMATIC_ACTIVITY = 3.0
NEMATIC_XI = 0.1


def _gel_velocity(s: SimpleNamespace) -> FreeModuleElement:
    """Return the velocity of the flow of a gel at a site, for the order the namespace holds."""
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


@register_system(
    field='physics/active_matter',
    tags=('field', 'active matter', 'nematic', 'tensor', 'liquid crystal', 'giant number fluctuations', 'pde'),
    references=('Marchetti et al. 2013', 'arXiv:1207.2929'),
    defaults={'until': 100.0, 'dt': 5.0, 'solver': RK4Solver(dt=0.02), 'backend': NumbaBackend()},
)
def dry_active_nematics(
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
) -> System:
    """Return a dry active nematic, a conserved density coupled to a symmetric traceless alignment tensor.

    On a periodic lattice the density and the tensor follow Section II.2 of Marchetti et al.,

        rho_t = -div(J),    J = -D grad(rho) - current div(Q),
        Q_t = H / gamma + beta (partial_i partial_j rho - delta_ij laplacian(rho) / 2),

    with the molecular field `H` of the Landau-de Gennes energy, whose coefficient `a (1 - rho / rho_c)` turns negative
    above the critical density `rho_c`. The current along `div(Q)` is the active current, which no free energy gives.

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

    Returns:
        system:
            The system, with the fields `nematic.rho` and `nematic.Q`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import dry_active_nematics
        >>> dry_active_nematics((8, 8)).elements[0].values['beta']
        0.5
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    sites = int(np.prod(shape))
    values = {
        'rho': density + perturbation * rng.standard_normal(sites),
        'Q': perturbation * rng.standard_normal((sites, 2)),
        'beta': beta,
    }
    nematic = FieldElement('nematic', element=NematicDensitySiteElement, space=LatticeSpace(shape), values=values)
    energy = LandauDeGennesEnergyInteraction(
        'energy',
        target='nematic',
        a=a,
        c=c,
        K=K,
        gamma=gamma,
        quadratic=lambda s: s.a * (1 - s.target.rho / critical_density),
    )

    def flux(s: SimpleNamespace) -> list[Expression]:
        """Return the current of the density, its diffusion and its active current."""
        active = tensor_divergence(s.target.Q)
        return [-D * item - current * other for item, other in zip(gradient(s.target.rho), active, strict=True)]

    transport = ConservationInteraction(
        'transport',
        target='nematic',
        state='rho',
        flux=flux,
        limiter=None,
        riemann=None,
    )
    return System(elements=[nematic], interactions=[energy, transport])


def _gel_settings(order: str = 'nematic', **_parameters: object) -> dict[str, object]:
    """Return the run settings of the polar gel, whose flow is slow, and none for the nematic, the catalog defaults."""
    return {'until': 200.0, 'dt': 1.0, 'solver': RK4Solver(dt=0.05)} if order == 'polar' else {}


@register_system(
    field='physics/active_matter',
    tags=('field', 'active matter', 'nematic', 'polar', 'tensor', 'gel', 'stokes flow', 'active turbulence', 'pde'),
    references=(
        'Marchetti et al. 2013',
        'arXiv:1207.2929',
        'Kruse et al. 2005',
        'Voituriez, Joanny, and Prost 2005',
        'Giomi et al. 2012',
        'arXiv:1110.4338',
    ),
    defaults={'until': 0.5, 'dt': 0.025, 'solver': RK4Solver(dt=5e-4)},
    settings=_gel_settings,
    observables=lambda system: [
        ObservableSpec('u_x', _velocity_x, group='gel', reduce=_sites),
        ObservableSpec('u_y', _velocity_y, group='gel', reduce=_sites),
    ],
)
def active_gels(
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
) -> System:
    """Return an active gel, a nematic tensor or a polarisation transported in the Stokes flow of its own stresses.

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
    holds the polarisation along the first axis at these two walls, with stress-free flow there.

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

    Returns:
        system:
            The system, with the field `gel.Q` or `gel.p`.

    Raises:
        ValidationError:
            The order is unknown, or a nematic is asked to lie in a film.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import active_gels
        >>> active_gels((8, 8), order='polar').elements[0].values['zeta']
        -0.5
        >>> active_gels(order='polar').elements[0].space.shape
        (32, 32)
    """
    if order not in GEL_ORDERS:
        error_msg = f'The order of the gel is {order!r}. Use one of {", ".join(GEL_ORDERS)}.'
        raise ValidationError(error_msg)
    if film and order != 'polar':
        error_msg = 'A film of the gel holds a polarisation at its walls. Use order=polar.'
        raise ValidationError(error_msg)
    polar = order == 'polar'
    shape = ((32, 32) if polar else (128, 128)) if shape is None else tuple(shape)
    sites = int(np.prod(shape))
    rng = np.random.default_rng(random_state)
    quadratic = (-1.0 if polar else (NEMATIC_CRITICAL_DENSITY - NEMATIC_DENSITY) / 2) if a is None else a
    cubic = (1.0 if polar else NEMATIC_DENSITY) if c is None else c
    activity = (-0.5 if polar else NEMATIC_ACTIVITY * NEMATIC_DENSITY**2) if zeta is None else zeta
    alignment = (0.0 if polar else NEMATIC_XI) if xi is None else xi
    distance = (1.0 if polar else NEMATIC_BOX / shape[0]) if spacing is None else spacing
    amplitude = (0.01 if polar else 0.1) if perturbation is None else perturbation
    noise = amplitude * rng.standard_normal((sites, 2))
    size = np.sqrt(-quadratic / cubic) if polar else np.sqrt(-2 * quadratic / cubic) / 2
    start = np.column_stack([np.full(sites, size), np.zeros(sites)]) + noise if quadratic < 0 else noise
    values = {
        'p' if polar else 'Q': start,
        'a': quadratic,
        'c': cubic,
        'K': K,
        'xi': alignment,
        'gamma': gamma,
        'eta': eta,
        'zeta': activity,
    }
    symmetry = {'p': ((None, 'even'), (None, 'odd'))} if film else None
    gel = FieldElement(
        'gel',
        element=PolarGelSiteElement if polar else NematicGelSiteElement,
        space=LatticeSpace(shape, spacing=distance),
        values=values,
        discretization='spectral',
        symmetry=symmetry,
    )
    return System(elements=[gel])
