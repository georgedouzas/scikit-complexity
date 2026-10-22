"""Continuum traffic models: the density and speed of traffic as fields along a road."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, parameter, state
from ....core.components import Element, Field
from ....core.laws import Flux, Rate, laplacian
from ....core.runtime import register_model
from ....core.solvers import SSPRK
from ....core.space import LatticeSpace
from ....core.system import System

KERNER_KONHAUSER_OFFSET = 3.72e-6

LighthillWhithamRichards = Element.define(
    'LighthillWhithamRichards',
    states={'rho': state(default=0.3, domain=Domain.NONNEGATIVE)},
    parameters={
        'vmax': parameter(default=1.0, domain=Domain.POSITIVE),
        'rho_max': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [Flux(s.rho, s.vmax * s.rho * (1 - s.rho / s.rho_max), riemann='godunov')],
)

Payne = Element.define(
    'Payne',
    states={'rho': state(default=0.3, domain=Domain.NONNEGATIVE), 'q': 0.21},
    parameters={
        'vmax': parameter(default=1.0, domain=Domain.POSITIVE),
        'rho_max': parameter(default=1.0, domain=Domain.POSITIVE),
        'tau': parameter(default=0.5, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Flux(s.rho, s.q),
        Flux(s.q, s.q**2 / s.rho + s.vmax * s.rho / (2 * s.tau * s.rho_max)),
        Rate(s.q, (s.rho * s.vmax * (1 - s.rho / s.rho_max) - s.q) / s.tau),
    ],
)

KernerKonhauser = Element.define(
    'KernerKonhauser',
    states={'rho': state(default=0.3, domain=Domain.NONNEGATIVE), 'q': 0.21},
    parameters={
        'V0': parameter(default=1.0, domain=Domain.POSITIVE),
        'rho_max': parameter(default=1.0, domain=Domain.POSITIVE),
        'tau': parameter(default=0.5, domain=Domain.POSITIVE),
        'c0': parameter(default=0.3, domain=Domain.POSITIVE),
        'mu': parameter(default=0.05, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Flux(s.rho, s.q),
        Flux(s.q, s.q**2 / s.rho + s.c0**2 * s.rho),
        Rate(
            s.q,
            (s.rho * s.V0 * (1 / (1 + exp((s.rho / s.rho_max - 0.25) / 0.06)) - KERNER_KONHAUSER_OFFSET) - s.q) / s.tau
            + s.mu * laplacian(s.q / s.rho),
        ),
    ],
)


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
    defaults={'until': 1.0, 'dt': 0.05, 'solver': SSPRK()},
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
    road = Field(
        'road',
        LighthillWhithamRichards,
        space,
        boundary={'rho': 'zero-flux'},
        values=values,
        discretization='finite-volume',
    )
    return System(components=[road])


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'field', 'conservation law', 'finite volume', 'pde'),
    references=('Payne 1971', 'Chowdhury, Santen, and Schadschneider 2000'),
    defaults={'until': 1.0, 'dt': 0.05, 'solver': SSPRK()},
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
    road = Field('road', Payne, LatticeSpace(n, spacing=1 / n), values=values, discretization='finite-volume')
    return System(components=[road])


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'field', 'conservation law', 'finite volume', 'jams', 'pde'),
    references=('Kerner and Konhäuser 1993', 'Chowdhury, Santen, and Schadschneider 2000'),
    defaults={'until': 100.0, 'dt': 0.2, 'solver': SSPRK(cfl=0.15)},
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
    road = Field('road', KernerKonhauser, space, values=values, discretization='finite-volume')
    return System(components=[road])
