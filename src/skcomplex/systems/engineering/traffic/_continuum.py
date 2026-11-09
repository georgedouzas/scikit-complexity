"""Continuum traffic systems: the density and speed of traffic as fields.

The fields lie along a road.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, erf, exp, oo, pi, sqrt, tanh
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import ConservationInteraction, RateInteraction
from ....core.components.laws.utils import laplacian, shifted
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import SSPRKSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.traffic import SpeedAdaptationInteraction


class GasKineticTrafficElement(Element):
    """Cell of a road with the density `rho` and the flow `Q` of the GKT system.

    The GKT system is the gas-kinetic-based traffic system. The density and the flow are conserved by conservation
    interactions, and the flow relaxes to a dynamic equilibrium speed that brakes on the density and the speed read
    ahead at the interaction point. Lengths are in kilometres and times in hours, so `T` and `tau` are fractions of an
    hour.

    Args:
        label:
            The label of the cell or the road.

        visual:
            How the cell is drawn, or `None` for `default_visual_`.

        rho:
            The initial density in vehicles per kilometre.

        Q:
            The initial flow in vehicles per hour.

        V0:
            The desired speed.

        T:
            The safe time headway.

        tau:
            The relaxation time of the speed.

        rho_max:
            The maximum density.

        gamma:
            The anticipation factor, the interaction distance in units of the safe distance `1 / rho_max + T V`.

        A0:
            The variance prefactor of free traffic.

        dA:
            The increase of the variance prefactor from free to congested traffic, halved.

        rho_c:
            The density of the transition of the variance prefactor.

        d_rho:
            The width of the transition of the variance prefactor.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import GasKineticTrafficElement
        >>> GasKineticTrafficElement('cell').gamma
        1.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 20.0,
        Q: Annotated[float, StateSpec()] = 1800.0,
        V0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 110.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.8 / 3600,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 35 / 3600,
        rho_max: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 160.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.2,
        A0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.008,
        dA: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        rho_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.27 * 160,
        d_rho: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05 * 160,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.Q = Q
        self.V0 = V0
        self.T = T
        self.tau = tau
        self.rho_max = rho_max
        self.gamma = gamma
        self.A0 = A0
        self.dA = dA
        self.rho_c = rho_c
        self.d_rho = d_rho


class KernerKonhauserElement(Element):
    """Cell of a road with the density and the flow of the Kerner-Konhäuser system.

    The density is `rho` and the flow is `q`. The density and the flow are conserved by conservation interactions, the
    flow relaxes to the equilibrium flow by a relaxation interaction, and the viscosity adds the rate `mu laplacian(q /
    rho)` to the flow.
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        q: Annotated[object, StateSpec()] = 0.21,
        V0: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rho_max: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        tau: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        c0: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        mu: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.q = q
        self.V0 = V0
        self.rho_max = rho_max
        self.tau = tau
        self.c0 = c0
        self.mu = mu


class LighthillWhithamRichardsElement(Element):
    """Cell of a road with the density `rho`, the free speed, and the jam density.

    The free speed is `vmax` and the jam density is `rho_max`.
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        vmax: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rho_max: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.vmax = vmax
        self.rho_max = rho_max


class PayneElement(Element):
    """Cell of a road with the density `rho` and the flow `q` of the Payne system.

    The density and the flow are conserved by conservation interactions, and the flow relaxes to the Greenshields flow
    by a relaxation interaction.
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        q: Annotated[object, StateSpec()] = 0.21,
        vmax: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rho_max: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        tau: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.q = q
        self.vmax = vmax
        self.rho_max = rho_max
        self.tau = tau


KERNER_KONHAUSER_OFFSET = 3.72e-6


def _get_variance_prefactor(road: SimpleNamespace, rho: object) -> Expression:
    """Return the variance prefactor of a road at the density `rho`.

    It is `A0 + dA (tanh((rho - rho_c) / d_rho) + 1)`.
    """
    prefactor: Expression = road.A0 + road.dA * (tanh((rho - road.rho_c) / road.d_rho) + 1)
    return prefactor


def _density_bump(n: int, density: float, bump: float) -> np.ndarray:
    """Return a uniform density on a ring with a smooth bump around the middle.

    The bump has the relative height `bump`.
    """
    centers = (np.arange(n) + 0.5) / n
    return density * (1 + bump * np.exp(-(((centers - 0.5) / 0.05) ** 2)))


class LighthillWhithamRichardsSystem(BaseSystem):
    """The Lighthill-Whitham-Richards system.

    It is the conservation of cars with the Greenshields speed law. The density follows `rho_t + (rho V(rho))_x = 0`
    with `V(rho) = vmax (1 - rho / rho_max)` on a road of length 1 with open edges. Free traffic at density `upstream`
    runs into a queue at density `downstream` in the second half of the road, and the shock between them moves at
    `vmax (1 - (upstream + downstream) / rho_max)`. The system has the density `road.rho`.

    Args:
        n:
            The number of cells.

        upstream:
            The density in the first half of the road.

        downstream:
            The density in the second half of the road.

        vmax:
            The speed on an empty road.

        rho_max:
            The jam density.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import LighthillWhithamRichardsSystem
        >>> float(LighthillWhithamRichardsSystem(10).prepare().system_.elements[0].values['rho'][-1])
        0.8
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'field', 'conservation law', 'finite volume', 'shocks', 'pde')
    references = ('Lighthill and Whitham 1955', 'Richards 1956', 'Chowdhury, Santen, and Schadschneider 2000')
    until = 1.0
    dt = 0.05
    solver = SSPRKSolver()

    def __init__(
        self,
        n: int = 200,
        *,
        upstream: float = 0.3,
        downstream: float = 0.8,
        vmax: float = 1.0,
        rho_max: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.upstream = upstream
        self.downstream = downstream
        self.vmax = vmax
        self.rho_max = rho_max

    @staticmethod
    def _get_lighthill_whitham_richards_flux(s: SimpleNamespace) -> Expression:
        """Return the Greenshields flux `vmax rho (1 - rho / rho_max)` of the density."""
        road = s.target
        return road.vmax * road.rho * (1 - road.rho / road.rho_max)

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        density = np.where(np.arange(self.n) < self.n // 2, self.upstream, self.downstream).astype(float)
        values = {'rho': density, 'vmax': self.vmax, 'rho_max': self.rho_max}
        space = LatticeSpace(self.n, spacing=1 / self.n)
        road = FieldElement(
            'road',
            element=LighthillWhithamRichardsElement,
            space=space,
            boundary={'rho': 'zero-flux'},
            values=values,
            discretization='finite-volume',
        )
        flux = self._get_lighthill_whitham_richards_flux
        density = ConservationInteraction('density', target='road', state='rho', flux=flux, riemann='godunov')
        return CompositeSystem(self.label, elements=[road], interactions=[density])


class PayneSystem(BaseSystem):
    """The Payne system, where the speed of traffic relaxes to the Greenshields speed.

    The speed also anticipates density. The density and the flow `q = rho v` on a ring road of length 1 follow
    `rho_t + q_x = 0` and `q_t + (q^2 / rho + P(rho))_x = (rho V(rho) - q) / τ`, with
    `V(rho) = vmax (1 - rho / rho_max)` and the anticipation pressure `P(rho) = vmax rho / (2 τ rho_max)`. This is the
    speed equation `v_t + v v_x = (V - v) / τ + V'(rho) rho_x / (2 τ rho)` in conservation form. The flow starts in
    equilibrium with a bump of density. The system has the density `road.rho` and the flow `road.q`.

    Args:
        n:
            The number of cells.

        density:
            The mean density.

        bump:
            The relative height of the density bump.

        vmax:
            The speed on an empty road.

        rho_max:
            The jam density.

        tau:
            The relaxation time of the speed.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import PayneSystem
        >>> PayneSystem(10).prepare().system_.elements[0].values['tau']
        0.5
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'field', 'conservation law', 'finite volume', 'pde')
    references = ('Payne 1971', 'Chowdhury, Santen, and Schadschneider 2000')
    until = 1.0
    dt = 0.05
    solver = SSPRKSolver()

    def __init__(
        self,
        n: int = 200,
        *,
        density: float = 0.3,
        bump: float = 0.2,
        vmax: float = 1.0,
        rho_max: float = 1.0,
        tau: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.density = density
        self.bump = bump
        self.vmax = vmax
        self.rho_max = rho_max
        self.tau = tau

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        density = self.density
        rho = _density_bump(self.n, density, self.bump)
        values = {
            'rho': rho,
            'q': rho * self.vmax * (1 - rho / self.rho_max),
            'vmax': self.vmax,
            'rho_max': self.rho_max,
            'tau': self.tau,
        }
        road = FieldElement(
            'road',
            element=PayneElement,
            space=LatticeSpace(self.n, spacing=1 / self.n),
            values=values,
            discretization='finite-volume',
        )
        density = ConservationInteraction('density', target='road', state='rho', flux=lambda s: s.target.q)
        flow = ConservationInteraction(
            'flow',
            target='road',
            state='q',
            flux=lambda s: s.target.q**2 / s.target.rho
            + s.target.vmax * s.target.rho / (2 * s.target.tau * s.target.rho_max),
        )
        relaxation = SpeedAdaptationInteraction(
            'relaxation',
            target='road',
            state='q',
            goal=lambda s: s.target.rho * s.target.vmax * (1 - s.target.rho / s.target.rho_max),
            time=lambda s: s.target.tau,
        )
        return CompositeSystem(self.label, elements=[road], interactions=[density, flow, relaxation])


class KernerKonhauserSystem(BaseSystem):
    """The Kerner-Konhäuser system, where viscous traffic forms moving clusters.

    The clusters are dense, and the traffic flow is viscous. The density and the flow `q = rho v` on a ring road follow
    `rho_t + q_x = 0` and `q_t + (q^2 / rho + c0^2 rho)_x = (rho V(rho) - q) / τ + μ v_xx`, with the equilibrium speed
    `V(rho) = V0 (1 / (1 + exp((rho / rho_max - 0.25) / 0.06)) - 3.72e-6)`. The flow starts in equilibrium with a bump
    of density, which grows into a cluster at intermediate densities. The system has the density `road.rho` and the flow
    `road.q`.

    Args:
        n:
            The number of cells.

        length:
            The length of the ring road.

        density:
            The mean density.

        bump:
            The relative height of the density bump.

        V0:
            The scale of the equilibrium speed.

        rho_max:
            The jam density.

        tau:
            The relaxation time of the speed.

        c0:
            The speed of sound of the traffic pressure `c0^2 rho`.

        mu:
            The viscosity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import KernerKonhauserSystem
        >>> KernerKonhauserSystem(10).prepare().system_.elements[0].values['mu']
        0.2
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'field', 'conservation law', 'finite volume', 'jams', 'pde')
    references = ('Kerner and Konhäuser 1993', 'Chowdhury, Santen, and Schadschneider 2000')
    until = 100.0
    dt = 0.2
    solver = SSPRKSolver(cfl=0.15)

    def __init__(
        self,
        n: int = 40,
        *,
        length: float = 10.0,
        density: float = 0.25,
        bump: float = 0.1,
        V0: float = 1.0,
        rho_max: float = 1.0,
        tau: float = 1.0,
        c0: float = 0.45,
        mu: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.length = length
        self.density = density
        self.bump = bump
        self.V0 = V0
        self.rho_max = rho_max
        self.tau = tau
        self.c0 = c0
        self.mu = mu

    @staticmethod
    def _get_kerner_konhauser_speed(road: SimpleNamespace) -> Expression:
        """Return the equilibrium speed of a road in the Kerner-Konhäuser system.

        It is `V0 (1 / (1 + exp((rho / rho_max - 0.25) / 0.06)) - 3.72e-6)`.
        """
        return road.V0 * (1 / (1 + exp((road.rho / road.rho_max - 0.25) / 0.06)) - KERNER_KONHAUSER_OFFSET)

    @staticmethod
    def _kerner_konhauser_speed(rho: np.ndarray, V0: float, rho_max: float) -> np.ndarray:
        """Return the equilibrium speed of the Kerner-Konhäuser system."""
        return V0 * (1 / (1 + np.exp((rho / rho_max - 0.25) / 0.06)) - KERNER_KONHAUSER_OFFSET)

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        density = self.density
        rho = _density_bump(self.n, density, self.bump)
        values = {
            'rho': rho,
            'q': rho * self._kerner_konhauser_speed(rho, self.V0, self.rho_max),
            'V0': self.V0,
            'rho_max': self.rho_max,
            'tau': self.tau,
            'c0': self.c0,
            'mu': self.mu,
        }
        space = LatticeSpace(self.n, spacing=self.length / self.n)
        road = FieldElement(
            'road',
            element=KernerKonhauserElement,
            space=space,
            values=values,
            discretization='finite-volume',
        )
        density = ConservationInteraction('density', target='road', state='rho', flux=lambda s: s.target.q)
        flow = ConservationInteraction(
            'flow',
            target='road',
            state='q',
            flux=lambda s: s.target.q**2 / s.target.rho + s.target.c0**2 * s.target.rho,
        )
        relaxation = SpeedAdaptationInteraction(
            'relaxation',
            target='road',
            state='q',
            goal=lambda s: s.target.rho * self._get_kerner_konhauser_speed(s.target),
            time=lambda s: s.target.tau,
        )
        viscosity = RateInteraction(
            'viscosity',
            target='road',
            state='q',
            rate=lambda s: s.target.mu * laplacian(s.target.q / s.target.rho),
        )
        return CompositeSystem(self.label, elements=[road], interactions=[density, flow, relaxation, viscosity])


class GktTrafficSystem(BaseSystem):
    """The gas-kinetic-based traffic system.

    Drivers brake on the traffic at an interaction point ahead. The density and the flow `Q = rho V` on a ring road
    follow `rho_t + Q_x = 0` and `Q_t + (Q**2 / rho + rho A(rho) V**2)_x = rho (Ve* - V) / tau`, where the dynamic
    equilibrium speed

        Ve* = V0 (1 - (theta + theta_a) / (2 A(rho_max)) (rho_a T / (1 - rho_a / rho_max))**2 B(dV))

    reads the density `rho_a` and the speed `V_a` at the interaction point `x + gamma (1 / rho_max + T V)`, with the
    variances `theta = A(rho) V**2`, the variance prefactor `A(rho) = A0 + dA (tanh((rho - rho_c) / d_rho) + 1)`, the
    dimensionless speed difference `dV = (V - V_a) / sqrt(theta + theta_a)`, and the Boltzmann factor
    `B(d) = 2 (d phi(d) + (1 + d**2) Phi(d))` of the standard normal density `phi` and distribution `Phi`. Lengths are
    in kilometres and times in hours, and the defaults are the calibrated values of Treiber, Hennecke, and Helbing. The
    flow starts in equilibrium with their dipole perturbation of the density, of height `amplitude`. Homogeneous
    traffic is unstable between about 24 and 51 vehicles per kilometre, where the perturbation grows into stop-and-go
    waves whose downstream fronts move against the traffic.

    The system has the density `road.rho` and the flow `road.Q`.

    Args:
        n:
            The number of cells.

        length:
            The length of the ring road in kilometres.

        density:
            The mean density in vehicles per kilometre.

        amplitude:
            The height of the dipole perturbation in vehicles per kilometre.

        V0:
            The desired speed in kilometres per hour.

        T:
            The safe time headway in hours.

        tau:
            The relaxation time of the speed in hours.

        rho_max:
            The maximum density in vehicles per kilometre.

        gamma:
            The anticipation factor.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import GktTrafficSystem
        >>> GktTrafficSystem(10).prepare().system_.elements[0].values['gamma']
        1.2
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'field', 'conservation law', 'finite volume', 'non-local', 'stop-and-go', 'pde')
    references = ('Treiber, Hennecke, and Helbing 1999', 'arXiv:cond-mat/0012229')
    until = 1.0
    dt = 1 / 60
    solver = SSPRKSolver(cfl=0.4)

    def __init__(
        self,
        n: int = 200,
        *,
        length: float = 10.0,
        density: float = 35.0,
        amplitude: float = 2.0,
        V0: float = 110.0,
        T: float = 1.8 / 3600,
        tau: float = 35 / 3600,
        rho_max: float = 160.0,
        gamma: float = 1.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.length = length
        self.density = density
        self.amplitude = amplitude
        self.V0 = V0
        self.T = T
        self.tau = tau
        self.rho_max = rho_max
        self.gamma = gamma

    @staticmethod
    def _get_gkt_momentum_flux(s: SimpleNamespace) -> Expression:
        """Return the flux of the flow.

        It is `Q**2 / rho` plus the traffic pressure `rho A(rho) V**2`.
        """
        road = s.target
        flux: Expression = (1 + _get_variance_prefactor(road, road.rho)) * road.Q**2 / road.rho
        return flux

    @staticmethod
    def _get_gkt_relaxation(s: SimpleNamespace) -> Expression:
        """Return the relaxation `rho (Ve* - V) / tau` of the flow.

        The flow relaxes to the dynamic equilibrium speed of the GKT system. The density and the speed are read at the
        interaction point `x + gamma (1 / rho_max + T V)`, and the braking term grows with the variances
        `theta = A(rho) V**2` of both points and with the Boltzmann factor of their dimensionless speed difference.
        """
        road = s.target
        speed = road.Q / road.rho
        ahead = [road.gamma * (1 / road.rho_max + road.T * speed)]
        density_ahead = shifted(road.rho, ahead)
        speed_ahead = shifted(road.Q / road.rho, ahead)
        variances = _get_variance_prefactor(road, road.rho) * speed**2 + _get_variance_prefactor(
            road,
            density_ahead,
        ) * (speed_ahead**2)
        difference = (speed - speed_ahead) / sqrt(variances)
        boltzmann = 2 * (
            difference * exp(-(difference**2) / 2) / sqrt(2 * pi)
            + (1 + difference**2) * (1 + erf(difference / sqrt(2))) / 2
        )
        gap = density_ahead * road.T / (1 - density_ahead / road.rho_max)
        braking = variances / (2 * _get_variance_prefactor(road, road.rho_max)) * gap**2 * boltzmann
        relaxation: Expression = road.rho * (road.V0 * (1 - braking) - speed) / road.tau
        return relaxation

    @staticmethod
    def _gkt_equilibrium_speed(rho: np.ndarray, road: GasKineticTrafficElement) -> np.ndarray:
        """Return the equilibrium speed of homogeneous GKT traffic.

        It is `Vt**2 / (2 V0) (sqrt(1 + 4 V0**2 / Vt**2) - 1)`.
        """
        prefactors = [
            road.A0 + road.dA * (np.tanh((value - road.rho_c) / road.d_rho) + 1) for value in (rho, road.rho_max)
        ]
        squared = prefactors[1] / prefactors[0] * ((1 / rho - 1 / road.rho_max) / road.T) ** 2
        return np.asarray(squared / (2 * road.V0) * (np.sqrt(1 + 4 * road.V0**2 / squared) - 1))

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        centers = (np.arange(self.n) + 0.5) * self.length / self.n
        start, width, wake = 0.3 * self.length, 0.20125, 0.805
        rho = self.density + self.amplitude * (
            np.cosh((centers - start) / width) ** -2 - width / wake * np.cosh((centers - start - 1.00625) / wake) ** -2
        )
        values = {'V0': self.V0, 'T': self.T, 'tau': self.tau, 'rho_max': self.rho_max, 'gamma': self.gamma}
        cell = GasKineticTrafficElement(
            'cell',
            V0=self.V0,
            T=self.T,
            tau=self.tau,
            rho_max=self.rho_max,
            gamma=self.gamma,
        )
        road = FieldElement(
            'road',
            element=GasKineticTrafficElement,
            space=LatticeSpace(self.n, spacing=self.length / self.n),
            values={'rho': rho, 'Q': rho * self._gkt_equilibrium_speed(rho, cell), **values},
            discretization='finite-volume',
        )
        vehicles = ConservationInteraction('density', target='road', state='rho', flux=lambda s: s.target.Q)
        momentum = ConservationInteraction(
            'flow',
            target='road',
            state='Q',
            flux=self._get_gkt_momentum_flux,
            source=self._get_gkt_relaxation,
        )
        return CompositeSystem(self.label, elements=[road], interactions=[vehicles, momentum])
