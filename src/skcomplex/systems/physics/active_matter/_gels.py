"""Active gels: contractile cells that drag a viscoelastic gel, whose velocity solves a force balance at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, sqrt

from ....core.components.elements import FieldElement
from ....core.components.interactions import AdvectionInteraction, DiffusionInteraction, RateInteraction
from ....core.components.laws.utils import inverse_helmholtz, partial
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._contractile_gel_site import ContractileGelSiteElement


def _gel_velocity(s: SimpleNamespace) -> Expression:
    """Return the gel velocity, the solution of `(gamma - eta d_xx) v = d_x (E d_x u + T0 c / (c + c_sat))`."""
    stress = s.E * partial(s.u, 0) + s.T0 * s.c / (s.c + s.c_sat)
    velocity: Expression = inverse_helmholtz(-partial(stress, 0) / s.eta, sqrt(s.gamma / s.eta))
    return velocity


def _transport(s: SimpleNamespace) -> Expression:
    """Return the velocity of a site of the target gel."""
    return _gel_velocity(s.target)


def _diffusion(s: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient of the cells."""
    diffusion: Expression = s.target.D
    return diffusion


def _sites(values: np.ndarray) -> np.ndarray:
    """Return the values of every site, the observed field itself."""
    return np.array(values)


@register_system(
    field='physics/active_matter',
    tags=('active gel', 'contractile', 'viscoelastic', 'field', 'inversion', 'pattern formation', 'pde'),
    references=('Brauns and Marchetti 2024', 'arXiv:2306.08868'),
    defaults={'until': 30.0, 'dt': 0.1, 'solver': RK4Solver(dt=0.005)},
    observables=lambda system: [ObservableSpec('v', _gel_velocity, group='gel', reduce=_sites)],
)
def active_viscoelastic_gel(
    n_sites: int = 128,
    *,
    length: float = 10.0,
    density: float = 1.0,
    perturbation: float = 0.01,
    D: float = 0.1,
    gamma: float = 1.0,
    eta: float = 1.0,
    E: float = 0.1,
    T0: float = 4.0,
    c_sat: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a one-dimensional active viscoelastic gel, where contractile cells concentrate by dragging the gel.

    On a periodic line, the density `c` of the cells and the displacement `u` of a Kelvin-Voigt gel follow

        c_t + (v c)_x = D c_xx,    u_t = v,    (gamma - eta d_xx) v = d_x (E u_x + Ta(c)),

    with the active stress `Ta(c) = T0 c / (c + c_sat)`. The force balance holds at every step, as the modified
    Helmholtz inversion of the stress gradient with the screening `sqrt(gamma / eta)`. Without stiffness the uniform
    state loses stability at long wavelengths above the contractile threshold `c Ta'(c) > gamma D`, and the density
    concentrates into peaks. The source leaves the active stress open, and the saturating one keeps the peaks finite.
    The catalog entry streams the velocity as the observable `v`.

    Args:
        n_sites:
            The number of sites.

        length:
            The length of the line.

        density:
            The mean density of the cells.

        perturbation:
            The standard deviation of the initial density around its mean.

        D:
            The diffusion coefficient of the cells.

        gamma:
            The friction of the gel with the substrate.

        eta:
            The viscosity of the gel.

        E:
            The stiffness of the gel.

        T0:
            The largest active stress of the cells.

        c_sat:
            The density at which the active stress is half its largest value.

        random_state:
            The seed or generator of the initial density.

    Returns:
        system:
            The system, with the density `gel.c` and the displacement `gel.u`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import active_viscoelastic_gel
        >>> active_viscoelastic_gel(16).elements[0].values['eta']
        1.0
    """
    rng = np.random.default_rng(random_state)
    values = {
        'c': density + perturbation * rng.standard_normal(n_sites),
        'u': 0.0,
        'D': D,
        'gamma': gamma,
        'eta': eta,
        'E': E,
        'T0': T0,
        'c_sat': c_sat,
    }
    gel = FieldElement(
        'gel',
        element=ContractileGelSiteElement,
        space=LatticeSpace(n_sites, spacing=length / n_sites),
        values=values,
    )
    transport = AdvectionInteraction('transport', target='gel', state='c', drift=_transport, limiter=None, riemann=None)
    diffusion = DiffusionInteraction('diffusion', target='gel', state='c', coefficient=_diffusion)
    displacement = RateInteraction('displacement', target='gel', state='u', rate=_transport)
    return System(elements=[gel], interactions=[transport, diffusion, displacement])
