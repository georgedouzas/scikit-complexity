"""Kuramoto models: phase oscillators coupled all to all, in populations, on rings, with delays, and as a mean field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import cos, sin

from ....core.base import Domain, Expression, VisualSpec, parameter, role, state
from ....core.model import (
    ContinuousSpace,
    Element,
    Interaction,
    Law,
    Rate,
    Symbols,
    System,
    delayed,
    mean,
)
from ....core.runtime import register_model
from ._utils import _order_parameter

PhaseOscillator = Element.define(
    'PhaseOscillator',
    states={'theta': 0.0},
    parameters={'omega': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)

DelayedCoupling = Interaction.define(
    'DelayedCoupling',
    roles=['source', 'target'],
    parameters={'K': 1.0, 'tau': parameter(default=1.0, domain=Domain.POSITIVE)},
    laws=lambda s: [Rate(s.target.theta, s.K * sin(delayed(s.source.theta, s.tau) - s.target.theta))],
)

SitedOscillator = Element.define(
    'SitedOscillator',
    states={'theta': 0.0, 'x': state(default=0.0, vector=True)},
    parameters={'omega': 0.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)

PhaseLag = Interaction.define(
    'PhaseLag',
    roles=['target'],
    parameters={'K': 1.0, 'alpha': 0.0},
    laws=lambda s: [Rate(s.target.theta, -s.K * sin(s.alpha))],
)


def _mean_field_rate(s: Symbols) -> Expression:
    """Return `K mean_j sin(theta_j - theta_i - alpha)`, written with the means of `cos theta_j` and `sin theta_j`."""
    shifted = s.target.theta + s.alpha
    return s.K * (mean(sin(s.source.theta)) * cos(shifted) - mean(cos(s.source.theta)) * sin(shifted))


MeanFieldCoupling = Interaction.define(
    'MeanFieldCoupling',
    roles=['target', 'source'],
    parameters={'K': 1.0, 'alpha': 0.0},
    laws=lambda s: [Rate(s.target.theta, _mean_field_rate(s))],
)

MeanFieldAmplitude = Element.define(
    'MeanFieldAmplitude',
    states={'x': 0.1, 'y': 0.0},
    parameters={
        'K': parameter(default=2.0, domain=Domain.NONNEGATIVE),
        'Delta': parameter(default=0.5, domain=Domain.NONNEGATIVE),
        'omega': 0.0,
    },
    laws=lambda s: [
        Rate(s.x, (-s.Delta + s.K / 2 * (1 - s.x**2 - s.y**2)) * s.x - s.omega * s.y),
        Rate(s.y, (-s.Delta + s.K / 2 * (1 - s.x**2 - s.y**2)) * s.y + s.omega * s.x),
    ],
)


def _spread_phases(size: int, seed: int) -> list[float]:
    """Return phases drawn uniformly from the circle with a seed."""
    return [float(phase) for phase in np.random.default_rng(seed).uniform(0, 2 * math.pi, size)]


class PhaseCoupling(Interaction):
    """Sine coupling with a phase lag between two phase oscillators, `theta_i' += K sin(theta_j - theta_i - alpha)`.

    Every pair of members closer than the cutoff couples both ways. The members carry the vector state `x`, their
    place, which decides the pairs.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group of the first members.

        second:
            The path of the group of the second members.

        cutoff:
            The distance beyond which pairs do not interact.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        K:
            The coupling strength of one pair.

        alpha:
            The phase lag.

    Examples:
        >>> from skcomplex.models.physics.synchronization import PhaseCoupling
        >>> PhaseCoupling('coupling', first='phases', second='phases', K=0.1).alpha
        0.0
    """

    over: ClassVar[str] = 'pairs'
    position: ClassVar[str] = 'x'
    first: str = role()
    second: str = role()
    cutoff: float = parameter(default=1.0, domain=Domain.POSITIVE)
    period: float | None = None

    K: float = parameter(default=1.0)
    alpha: float = parameter(default=0.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the sine coupling of both members of a pair.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        first, second = s.first.theta, s.second.theta
        return [
            Rate(first, s.K * sin(second - first - s.alpha)),
            Rate(second, s.K * sin(first - second - s.alpha)),
        ]


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'delay', 'oscillators'),
    references=('Yeung and Strogatz 1999', 'Kuramoto 1984'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def delayed_kuramoto(
    n_oscillators: int = 5,
    *,
    K: float = 2.0,
    tau: float = 1.0,
    spread: float = 0.2,
    seed: int = 0,
) -> System:
    """Return Kuramoto phase oscillators coupled all to all through signals that arrive after a delay.

    Oscillator `i` follows `theta_i' = omega_i + (K / N) sum_j sin(theta_j(t - tau) - theta_i)`, with natural
    frequencies spread evenly around 1 and initial phases spread around the circle.

    Args:
        n_oscillators:
            The number of oscillators.

        K:
            The coupling strength.

        tau:
            The delay of the coupling.

        spread:
            The width of the interval of natural frequencies around 1.

        seed:
            The seed of the initial phases.

    Returns:
        system:
            The model, with the phases `oscillator_<i>.theta`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import delayed_kuramoto
        >>> len(delayed_kuramoto(n_oscillators=3).interactions)
        6
    """
    phases = [2 * math.pi * ((seed * 0.618 + index / n_oscillators) % 1.0) for index in range(n_oscillators)]
    omegas = [1.0 + spread * (index / max(n_oscillators - 1, 1) - 0.5) for index in range(n_oscillators)]
    elements = [
        PhaseOscillator(f'oscillator_{index}', theta=phases[index], omega=omegas[index])
        for index in range(n_oscillators)
    ]
    interactions = [
        DelayedCoupling(
            f'coupling_{source}_{target}',
            source=f'oscillator_{source}',
            target=f'oscillator_{target}',
            K=K / n_oscillators,
            tau=tau,
        )
        for source in range(n_oscillators)
        for target in range(n_oscillators)
        if source != target
    ]
    return System(components=elements, interactions=interactions)


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'phase', 'mean field'),
    references=('Kuramoto 1984', 'Datseris and Parlitz 2022', 'Gros 2015', 'Jensen 2023'),
    defaults={'until': 50.0, 'dt': 0.5},
    observables=lambda system: _order_parameter('order', 'oscillators'),
)
def kuramoto(n_oscillators: int = 50, *, K: float = 4.0, sigma: float = 1.0, seed: int = 0) -> System:
    """Return Kuramoto oscillators coupled all to all, `theta_i' = omega_i + (K / N) sum_j sin(theta_j - theta_i)`.

    Every oscillator reads the group means of `cos theta` and `sin theta`, so the coupling costs time linear in the
    number of oscillators. The natural frequencies are drawn from a normal distribution of width `sigma`, and the
    phases uniformly from the circle. Above the critical coupling `K_c = 2 sigma sqrt(2 / pi)` a synchronized cluster
    forms, and the order parameter `r`, the modulus of the mean of `exp(i theta)`, grows from about `1 / sqrt(N)`
    towards 1.

    Args:
        n_oscillators:
            The number of oscillators.

        K:
            The coupling strength.

        sigma:
            The standard deviation of the natural frequencies.

        seed:
            The seed of the natural frequencies and the initial phases.

    Returns:
        system:
            The model, with the phases `oscillators.theta` and the order parameter `order`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import kuramoto
        >>> system = kuramoto(10)
        >>> system.elements[0].size, system.space
        (10, None)
    """
    rng = np.random.default_rng(seed)
    oscillators = PhaseOscillator.group(
        'oscillators',
        n_oscillators,
        theta=_spread_phases(n_oscillators, seed),
        omega=rng.normal(0.0, sigma, n_oscillators),
    )
    coupling = MeanFieldCoupling('coupling', target='oscillators', source='oscillators', K=K)
    return System(
        components=[oscillators],
        interactions=[coupling],
    )


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'phase', 'chimera'),
    references=('Abrams et al. 2008', 'Jensen 2023'),
    defaults={'until': 100.0, 'dt': 1.0},
    observables=lambda system: [*_order_parameter('first_order', 'first'), *_order_parameter('second_order', 'second')],
)
def kuramoto_sakaguchi_chimera(
    n_oscillators: int = 32,
    *,
    A: float = 0.2,
    beta: float = 0.1,
    spread: float = 0.1,
    seed: int = 0,
) -> System:
    """Return two populations of identical phase oscillators with phase lag, where one synchronizes and one drifts.

    Oscillator `i` of population `p` follows `theta_i' = sum_q (K_pq / N) sum_j sin(theta_j - theta_i - alpha)`, with
    the stronger coupling `mu = (1 + A) / 2` inside a population, the weaker `nu = (1 - A) / 2` between them, and the
    lag `alpha = pi / 2 - beta`. Every population reads the means of `cos theta` and `sin theta` of both populations.
    The first population starts nearly synchronized and the second spread over the circle. The populations settle in a
    chimera, where one is fully synchronized and the other keeps drifting with a partial order parameter, which
    breathes for `A = 0.2`. Which population ends up coherent depends on the initial phases.

    Args:
        n_oscillators:
            The number of oscillators of every population.

        A:
            The disparity of the couplings inside and between the populations.

        beta:
            The distance of the phase lag from `pi / 2`.

        spread:
            The width of the initial phases of the first population.

        seed:
            The seed of the initial phases.

    Returns:
        system:
            The model, with the phases `first.theta` and `second.theta` and the order parameters `first_order` and
            `second_order`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import kuramoto_sakaguchi_chimera
        >>> len(kuramoto_sakaguchi_chimera(4).interactions)
        4
    """
    rng = np.random.default_rng(seed)
    alpha, mu, nu = math.pi / 2 - beta, (1 + A) / 2, (1 - A) / 2
    populations = [
        PhaseOscillator.group(
            'first',
            n_oscillators,
            theta=rng.normal(0.0, spread, n_oscillators),
            omega=0.0,
            visual=VisualSpec('phase_ring', {'phase': 'theta'}),
        ),
        PhaseOscillator.group(
            'second',
            n_oscillators,
            theta=_spread_phases(n_oscillators, seed + 1),
            omega=0.0,
            visual=VisualSpec('phase_ring', {'phase': 'theta'}),
        ),
    ]
    interactions: list[Interaction] = [
        MeanFieldCoupling(
            f'{target}_from_{source}',
            target=target,
            source=source,
            K=mu if target == source else nu,
            alpha=alpha,
        )
        for target in ('first', 'second')
        for source in ('first', 'second')
    ]
    return System(
        components=populations,
        interactions=interactions,
    )


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'phase', 'chimera', 'nonlocal coupling'),
    references=('Kuramoto and Battogtokh 2002', 'Datseris and Parlitz 2022'),
    defaults={'until': 100.0, 'dt': 1.0},
    observables=lambda system: _order_parameter('order', 'ring'),
)
def nonlocal_ring_chimera(
    n_oscillators: int = 100,
    *,
    R: int = 14,
    alpha: float = 1.46,
    omega: float = 0.0,
    seed: int = 0,
) -> System:
    """Return identical phase oscillators on a ring, each coupled to its `R` nearest neighbours on both sides.

    Oscillator `i` follows `phi_i' = omega - (1 / 2R) sum_{j = i - R}^{i + R} sin(phi_i - phi_j + alpha)`, where the sum
    includes the oscillator itself. Near `alpha = pi / 2`, coherent and incoherent domains coexist on the ring as a
    chimera state, which is a long transient on small rings.

    Args:
        n_oscillators:
            The number of oscillators on the ring.

        R:
            The range of the coupling.

        alpha:
            The phase lag.

        omega:
            The natural frequency.

        seed:
            The seed of the initial phases.

    Returns:
        system:
            The model, with the phases `ring.theta` and the places of the oscillators `ring.x`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import nonlocal_ring_chimera
        >>> nonlocal_ring_chimera(20, R=3).interactions[0].cutoff
        3.5
    """
    weight = 1 / (2 * R)
    ring = SitedOscillator.group(
        'ring',
        n_oscillators,
        theta=_spread_phases(n_oscillators, seed),
        omega=omega,
        x=np.arange(n_oscillators, dtype=float).reshape(-1, 1),
        visual=VisualSpec('phase_ring', {'phase': 'theta', 'position': 'x'}),
    )
    return System(
        components=[ring],
        interactions=[
            PhaseCoupling(
                'coupling',
                first='ring',
                second='ring',
                cutoff=R + 0.5,
                period=float(n_oscillators),
                K=weight,
                alpha=alpha,
            ),
            PhaseLag('lag', target='ring', K=weight, alpha=alpha),
        ],
        space=ContinuousSpace(n_dim=1),
    )


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'ode', 'mean field', 'dimension reduction'),
    references=('Ott and Antonsen 2008', 'Bick et al. 2020'),
    defaults={'until': 30.0, 'dt': 0.1},
)
def ott_antonsen(
    r: float = 0.1,
    *,
    K: float = 2.0,
    Delta: float = 0.5,
    omega: float = 0.0,
) -> System:
    """Return the Ott-Antonsen equation for the order parameter of infinitely many Kuramoto oscillators.

    For natural frequencies with a Lorentzian distribution of center `omega` and half width `Delta`, the complex order
    parameter `z = x + i y` follows exactly `z' = (i omega - Delta) z + (K / 2)(1 - |z|²) z`. Its modulus settles at
    `sqrt(1 - 2 Delta / K)` above the critical coupling `K_c = 2 Delta`, and at 0 below it.

    Args:
        r:
            The initial modulus of the order parameter.

        K:
            The coupling strength.

        Delta:
            The half width of the distribution of natural frequencies.

        omega:
            The center of the distribution of natural frequencies.

    Returns:
        system:
            The model, with the real part `field.x` and the imaginary part `field.y` of the order parameter.

    Examples:
        >>> from skcomplex.models.physics.synchronization import ott_antonsen
        >>> ott_antonsen(K=3.0).elements[0].K
        3.0
    """
    return System(components=[MeanFieldAmplitude('field', x=r, y=0.0, K=K, Delta=Delta, omega=omega)])
