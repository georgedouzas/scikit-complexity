"""Continuum traffic models: the density and speed of traffic as fields along a road."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, erf, exp, pi, sqrt, tanh

from ....core.elements import FieldElement
from ....core.interactions import ConservationInteraction, RateInteraction, RelaxationInteraction
from ....core.laws.utils import laplacian, shifted
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import SSPRKSolver
from ..._catalog import register_model
from ._gas_kinetic_traffic import GasKineticTrafficElement
from ._kerner_konhauser import KernerKonhauserElement
from ._lighthill_whitham_richards import LighthillWhithamRichardsElement
from ._payne import PayneElement

KERNER_KONHAUSER_OFFSET = 3.72e-6


def _get_lighthill_whitham_richards_flux(s: SimpleNamespace) -> Expression:
    """Return the Greenshields flux `vmax rho (1 - rho / rho_max)` of the density."""
    road = s.target
    return road.vmax * road.rho * (1 - road.rho / road.rho_max)


def _get_kerner_konhauser_speed(road: SimpleNamespace) -> Expression:
    """Return the equilibrium speed `V0 (1 / (1 + exp((rho / rho_max - 0.25) / 0.06)) - 3.72e-6)` of a road."""
    return road.V0 * (1 / (1 + exp((road.rho / road.rho_max - 0.25) / 0.06)) - KERNER_KONHAUSER_OFFSET)


def _get_variance_prefactor(road: SimpleNamespace, rho: object) -> Expression:
    """Return the variance prefactor `A0 + dA (tanh((rho - rho_c) / d_rho) + 1)` of a road at the density `rho`."""
    prefactor: Expression = road.A0 + road.dA * (tanh((rho - road.rho_c) / road.d_rho) + 1)
    return prefactor


def _get_gkt_momentum_flux(s: SimpleNamespace) -> Expression:
    """Return the flux of the flow, `Q**2 / rho` plus the traffic pressure `rho A(rho) V**2`."""
    road = s.target
    flux: Expression = (1 + _get_variance_prefactor(road, road.rho)) * road.Q**2 / road.rho
    return flux


def _get_gkt_relaxation(s: SimpleNamespace) -> Expression:
    """Return the relaxation `rho (Ve* - V) / tau` of the flow to the dynamic equilibrium speed of the GKT model.

    The density and the speed are read at the interaction point `x + gamma (1 / rho_max + T V)`, and the braking term
    grows with the variances `theta = A(rho) V**2` of both points and with the Boltzmann factor of their dimensionless
    speed difference.
    """
    road = s.target
    speed = road.Q / road.rho
    ahead = [road.gamma * (1 / road.rho_max + road.T * speed)]
    density_ahead = shifted(road.rho, ahead)
    speed_ahead = shifted(road.Q / road.rho, ahead)
    variances = _get_variance_prefactor(road, road.rho) * speed**2 + _get_variance_prefactor(road, density_ahead) * (
        speed_ahead**2
    )
    difference = (speed - speed_ahead) / sqrt(variances)
    boltzmann = 2 * (
        difference * exp(-(difference**2) / 2) / sqrt(2 * pi)
        + (1 + difference**2) * (1 + erf(difference / sqrt(2))) / 2
    )
    gap = density_ahead * road.T / (1 - density_ahead / road.rho_max)
    braking = variances / (2 * _get_variance_prefactor(road, road.rho_max)) * gap**2 * boltzmann
    relaxation: Expression = road.rho * (road.V0 * (1 - braking) - speed) / road.tau
    return relaxation


def _gkt_equilibrium_speed(rho: np.ndarray, road: GasKineticTrafficElement) -> np.ndarray:
    """Return the equilibrium speed `Vt**2 / (2 V0) (sqrt(1 + 4 V0**2 / Vt**2) - 1)` of homogeneous GKT traffic."""
    prefactors = [road.A0 + road.dA * (np.tanh((value - road.rho_c) / road.d_rho) + 1) for value in (rho, road.rho_max)]
    squared = prefactors[1] / prefactors[0] * ((1 / rho - 1 / road.rho_max) / road.T) ** 2
    return np.asarray(squared / (2 * road.V0) * (np.sqrt(1 + 4 * road.V0**2 / squared) - 1))


def _density_bump(n: int, density: float, bump: float) -> np.ndarray:
    """Return a uniform density on a ring with a smooth bump of relative height `bump` around the middle."""
    centers = (np.arange(n) + 0.5) / n
    return density * (1 + bump * np.exp(-(((centers - 0.5) / 0.05) ** 2)))


def _kerner_konhauser_speed(rho: np.ndarray, V0: float, rho_max: float) -> np.ndarray:
    """Return the equilibrium speed of the Kerner-Konhäuser model."""
    return V0 * (1 / (1 + np.exp((rho / rho_max - 0.25) / 0.06)) - KERNER_KONHAUSER_OFFSET)


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'field', 'conservation law', 'finite volume', 'shocks', 'pde'),
    references=('Lighthill and Whitham 1955', 'Richards 1956', 'Chowdhury, Santen, and Schadschneider 2000'),
    defaults={'until': 1.0, 'dt': 0.05, 'solver': SSPRKSolver()},
)
def lighthill_whitham_richards(
    n: int = 200,
    *,
    upstream: float = 0.3,
    downstream: float = 0.8,
    vmax: float = 1.0,
    rho_max: float = 1.0,
) -> System:
    """Return the Lighthill-Whitham-Richards model, the conservation of cars with the Greenshields speed law.

    The density follows `rho_t + (rho V(rho))_x = 0` with `V(rho) = vmax (1 - rho / rho_max)` on a road of length 1
    with open edges. Free traffic at density `upstream` runs into a queue at density `downstream` in the second half of
    the road, and the shock between them moves at `vmax (1 - (upstream + downstream) / rho_max)`.

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

    Returns:
        system:
            The model, with the density `road.rho`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import lighthill_whitham_richards
        >>> float(lighthill_whitham_richards(10).elements[0].values['rho'][-1])
        0.8
    """
    density = np.where(np.arange(n) < n // 2, upstream, downstream).astype(float)
    values = {'rho': density, 'vmax': vmax, 'rho_max': rho_max}
    space = LatticeSpace(n, spacing=1 / n)
    road = FieldElement(
        'road',
        element=LighthillWhithamRichardsElement,
        space=space,
        boundary={'rho': 'zero-flux'},
        values=values,
        discretization='finite-volume',
    )
    flux = _get_lighthill_whitham_richards_flux
    density = ConservationInteraction('density', target='road', state='rho', flux=flux, riemann='godunov')
    return System(elements=[road], interactions=[density])


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'field', 'conservation law', 'finite volume', 'pde'),
    references=('Payne 1971', 'Chowdhury, Santen, and Schadschneider 2000'),
    defaults={'until': 1.0, 'dt': 0.05, 'solver': SSPRKSolver()},
)
def payne(
    n: int = 200,
    *,
    density: float = 0.3,
    bump: float = 0.2,
    vmax: float = 1.0,
    rho_max: float = 1.0,
    tau: float = 0.5,
) -> System:
    """Return the Payne model, where the speed of traffic relaxes to the Greenshields speed and anticipates density.

    The density and the flow `q = rho v` on a ring road of length 1 follow `rho_t + q_x = 0` and
    `q_t + (q^2 / rho + P(rho))_x = (rho V(rho) - q) / τ`, with `V(rho) = vmax (1 - rho / rho_max)` and the
    anticipation pressure `P(rho) = vmax rho / (2 τ rho_max)`. This is the speed equation
    `v_t + v v_x = (V - v) / τ + V'(rho) rho_x / (2 τ rho)` in conservation form. The flow starts in equilibrium
    with a bump of density.

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

    Returns:
        system:
            The model, with the density `road.rho` and the flow `road.q`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import payne
        >>> payne(10).elements[0].values['tau']
        0.5
    """
    rho = _density_bump(n, density, bump)
    values = {'rho': rho, 'q': rho * vmax * (1 - rho / rho_max), 'vmax': vmax, 'rho_max': rho_max, 'tau': tau}
    road = FieldElement(
        'road',
        element=PayneElement,
        space=LatticeSpace(n, spacing=1 / n),
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
    relaxation = RelaxationInteraction(
        'relaxation',
        target='road',
        state='q',
        goal=lambda s: s.target.rho * s.target.vmax * (1 - s.target.rho / s.target.rho_max),
        time=lambda s: s.target.tau,
    )
    return System(elements=[road], interactions=[density, flow, relaxation])


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'field', 'conservation law', 'finite volume', 'jams', 'pde'),
    references=('Kerner and Konhäuser 1993', 'Chowdhury, Santen, and Schadschneider 2000'),
    defaults={'until': 100.0, 'dt': 0.2, 'solver': SSPRKSolver(cfl=0.15)},
)
def kerner_konhauser(
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
) -> System:
    """Return the Kerner-Konhäuser model, where viscous traffic flow forms dense moving clusters.

    The density and the flow `q = rho v` on a ring road follow `rho_t + q_x = 0` and
    `q_t + (q^2 / rho + c0^2 rho)_x = (rho V(rho) - q) / τ + μ v_xx`, with the equilibrium speed
    `V(rho) = V0 (1 / (1 + exp((rho / rho_max - 0.25) / 0.06)) - 3.72e-6)`. The flow starts in equilibrium with a
    bump of density, which grows into a cluster at intermediate densities.

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

    Returns:
        system:
            The model, with the density `road.rho` and the flow `road.q`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import kerner_konhauser
        >>> kerner_konhauser(10).elements[0].values['mu']
        0.2
    """
    rho = _density_bump(n, density, bump)
    values = {
        'rho': rho,
        'q': rho * _kerner_konhauser_speed(rho, V0, rho_max),
        'V0': V0,
        'rho_max': rho_max,
        'tau': tau,
        'c0': c0,
        'mu': mu,
    }
    space = LatticeSpace(n, spacing=length / n)
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
    relaxation = RelaxationInteraction(
        'relaxation',
        target='road',
        state='q',
        goal=lambda s: s.target.rho * _get_kerner_konhauser_speed(s.target),
        time=lambda s: s.target.tau,
    )
    viscosity = RateInteraction(
        'viscosity',
        target='road',
        state='q',
        rate=lambda s: s.target.mu * laplacian(s.target.q / s.target.rho),
    )
    return System(elements=[road], interactions=[density, flow, relaxation, viscosity])


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'field', 'conservation law', 'finite volume', 'non-local', 'stop-and-go', 'pde'),
    references=('Treiber, Hennecke, and Helbing 1999', 'arXiv:cond-mat/0012229'),
    defaults={'until': 1.0, 'dt': 1 / 60, 'solver': SSPRKSolver(cfl=0.4)},
)
def gkt_traffic(
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
) -> System:
    """Return the gas-kinetic-based traffic model, where drivers brake on the traffic at an interaction point ahead.

    The density and the flow `Q = rho V` on a ring road follow `rho_t + Q_x = 0` and
    `Q_t + (Q**2 / rho + rho A(rho) V**2)_x = rho (Ve* - V) / tau`, where the dynamic equilibrium speed

        Ve* = V0 (1 - (theta + theta_a) / (2 A(rho_max)) (rho_a T / (1 - rho_a / rho_max))**2 B(dV))

    reads the density `rho_a` and the speed `V_a` at the interaction point `x + gamma (1 / rho_max + T V)`, with the
    variances `theta = A(rho) V**2`, the variance prefactor `A(rho) = A0 + dA (tanh((rho - rho_c) / d_rho) + 1)`, the
    dimensionless speed difference `dV = (V - V_a) / sqrt(theta + theta_a)`, and the Boltzmann factor
    `B(d) = 2 (d phi(d) + (1 + d**2) Phi(d))` of the standard normal density `phi` and distribution `Phi`. Lengths are
    in kilometres and times in hours, and the defaults are the calibrated values of Treiber, Hennecke, and Helbing. The
    flow starts in equilibrium with their dipole perturbation of the density, of height `amplitude`. Homogeneous
    traffic is unstable between about 24 and 51 vehicles per kilometre, where the perturbation grows into stop-and-go
    waves whose downstream fronts move against the traffic.

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

    Returns:
        system:
            The model, with the density `road.rho` and the flow `road.Q`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import gkt_traffic
        >>> gkt_traffic(10).elements[0].values['gamma']
        1.2
    """
    centers = (np.arange(n) + 0.5) * length / n
    start, width, wake = 0.3 * length, 0.20125, 0.805
    rho = density + amplitude * (
        np.cosh((centers - start) / width) ** -2 - width / wake * np.cosh((centers - start - 1.00625) / wake) ** -2
    )
    values = {'V0': V0, 'T': T, 'tau': tau, 'rho_max': rho_max, 'gamma': gamma}
    cell = GasKineticTrafficElement('cell', V0=V0, T=T, tau=tau, rho_max=rho_max, gamma=gamma)
    road = FieldElement(
        'road',
        element=GasKineticTrafficElement,
        space=LatticeSpace(n, spacing=length / n),
        values={'rho': rho, 'Q': rho * _gkt_equilibrium_speed(rho, cell), **values},
        discretization='finite-volume',
    )
    vehicles = ConservationInteraction('density', target='road', state='rho', flux=lambda s: s.target.Q)
    momentum = ConservationInteraction(
        'flow',
        target='road',
        state='Q',
        flux=_get_gkt_momentum_flux,
        source=_get_gkt_relaxation,
    )
    return System(elements=[road], interactions=[vehicles, momentum])
