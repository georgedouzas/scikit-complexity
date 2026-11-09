"""Active gels.

Contractile cells drag a viscoelastic gel, whose velocity solves a force balance at every step.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import inverse_helmholtz, partial
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import RK4Solver
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..common.fields import AdvectionInteraction, DiffusivityInteraction


class ContractileGelSiteElement(Element):
    """Site of a one-dimensional active viscoelastic gel.

    The site has the density of contractile cells and the gel displacement.

    The site holds the density `c` of the cells and the displacement `u` of a Kelvin-Voigt gel of stiffness `E` and
    viscosity `eta`, with friction `gamma` against a rigid substrate. The cells diffuse with the coefficient `D` and
    exert the active stress `T0 c / (c + c_sat)`.

    Args:
        label:
            The label of the site or the gel.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        c:
            The initial density of the cells.

        u:
            The initial displacement of the gel.

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

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ContractileGelSiteElement
        >>> ContractileGelSiteElement('site').T0
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        E: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        T0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        c_sat: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.u = u
        self.D = D
        self.gamma = gamma
        self.eta = eta
        self.E = E
        self.T0 = T0
        self.c_sat = c_sat


def _gel_velocity(s: SimpleNamespace) -> Expression:
    """Return the gel velocity.

    The velocity solves `(gamma - eta d_xx) v = d_x (E d_x u + T0 c / (c + c_sat))`.
    """
    stress = s.E * partial(s.u, 0) + s.T0 * s.c / (s.c + s.c_sat)
    velocity: Expression = inverse_helmholtz(-partial(stress, 0) / s.eta, sqrt(s.gamma / s.eta))
    return velocity


def _sites(values: np.ndarray) -> np.ndarray:
    """Return the values of every site, the observed field itself."""
    return np.array(values)


class ActiveViscoelasticGelSystem(BaseSystem):
    """A one-dimensional active viscoelastic gel.

    Contractile cells concentrate by dragging the gel.

    On a periodic line, the density `c` of the cells and the displacement `u` of a Kelvin-Voigt gel follow

        c_t + (v c)_x = D c_xx,    u_t = v,    (gamma - eta d_xx) v = d_x (E u_x + Ta(c)),

    with the active stress `Ta(c) = T0 c / (c + c_sat)`. The force balance holds at every step, as the modified
    Helmholtz inversion of the stress gradient with the screening `sqrt(gamma / eta)`. Without stiffness the uniform
    state loses stability at long wavelengths above the contractile threshold `c Ta'(c) > gamma D`, and the density
    concentrates into peaks. The source leaves the active stress open, and the saturating one keeps the peaks finite.
    The catalog entry streams the velocity as the observable `v`. The system has the density `gel.c` and the
    displacement `gel.u`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveViscoelasticGelSystem
        >>> ActiveViscoelasticGelSystem(16).prepare().system_.elements[0].values['eta']
        1.0
    """

    field = 'physics/active_matter'
    topics = ('active gel', 'contractile', 'viscoelastic', 'field', 'inversion', 'pattern formation', 'pde')
    references = ('Brauns and Marchetti 2024', 'arXiv:2306.08868')
    until = 30.0
    dt = 0.1
    solver = RK4Solver(dt=0.005)
    observables = (ObservableSpec('v', _gel_velocity, group='gel', reduce=_sites),)

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.length = length
        self.density = density
        self.perturbation = perturbation
        self.D = D
        self.gamma = gamma
        self.eta = eta
        self.E = E
        self.T0 = T0
        self.c_sat = c_sat
        self.random_state = random_state

    @staticmethod
    def _transport(s: SimpleNamespace) -> Expression:
        """Return the velocity of a site of the target gel."""
        return _gel_velocity(s.target)

    @staticmethod
    def _diffusion(s: SimpleNamespace) -> Expression:
        """Return the diffusion coefficient of the cells."""
        diffusion: Expression = s.target.D
        return diffusion

    def _build_system(self) -> CompositeSystem:
        """Return a one-dimensional active viscoelastic gel.

        Contractile cells concentrate by dragging the gel.
        """
        rng = np.random.default_rng(self.random_state)
        values = {
            'c': self.density + self.perturbation * rng.standard_normal(self.n_sites),
            'u': 0.0,
            'D': self.D,
            'gamma': self.gamma,
            'eta': self.eta,
            'E': self.E,
            'T0': self.T0,
            'c_sat': self.c_sat,
        }
        gel = FieldElement(
            'gel',
            element=ContractileGelSiteElement,
            space=LatticeSpace(self.n_sites, spacing=self.length / self.n_sites),
            values=values,
        )
        transport = AdvectionInteraction(
            'transport',
            target='gel',
            state='c',
            drift=self._transport,
            limiter=None,
            riemann=None,
        )
        diffusion = DiffusivityInteraction('diffusion', target='gel', state='c', coefficient=self._diffusion)
        displacement = RateInteraction('displacement', target='gel', state='u', rate=self._transport)
        return CompositeSystem(self.label, elements=[gel], interactions=[transport, diffusion, displacement])
