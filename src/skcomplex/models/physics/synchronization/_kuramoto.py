"""Kuramoto models: phase oscillators coupled all to all, in populations, on rings, with delays, and as a mean field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import BaseInteraction, Interaction, PairInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import delayed, mean
from ....core.spaces import ContinuousSpace
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model
from ._utils import _order_parameter


class PhaseOscillatorElement(Element):
    """Phase oscillator element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec()] = 0.0,
        omega: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the phase oscillator element."""
        return [RateLaw(s.theta, s.omega)]


class DelayedCouplingInteraction(Interaction):
    """Delayed coupling interaction."""

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[object, RoleSpec()],
        target: Annotated[object, RoleSpec()],
        K: Annotated[object, ParameterSpec()] = 1.0,
        tau: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.K = K
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the delayed coupling interaction."""
        return [RateLaw(s.target.theta, s.K * sin(delayed(s.source.theta, s.tau) - s.target.theta))]


class SitedOscillatorElement(Element):
    """Sited oscillator element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec()] = 0.0,
        x: Annotated[object, StateSpec(vector=True)] = 0.0,
        omega: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.x = x
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the sited oscillator element."""
        return [RateLaw(s.theta, s.omega)]


class PhaseLagInteraction(Interaction):
    """Phase lag interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        K: Annotated[object, ParameterSpec()] = 1.0,
        alpha: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.K = K
        self.alpha = alpha

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the phase lag interaction."""
        return [RateLaw(s.target.theta, -s.K * sin(s.alpha))]


def _mean_field_rate(s: SimpleNamespace) -> Expression:
    """Return `K mean_j sin(theta_j - theta_i - alpha)`, written with the means of `cos theta_j` and `sin theta_j`."""
    shifted = s.target.theta + s.alpha
    return s.K * (mean(sin(s.source.theta)) * cos(shifted) - mean(cos(s.source.theta)) * sin(shifted))


class MeanFieldCouplingInteraction(Interaction):
    """Mean field coupling interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        source: Annotated[object, RoleSpec()],
        K: Annotated[object, ParameterSpec()] = 1.0,
        alpha: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.K = K
        self.alpha = alpha

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the mean field coupling interaction."""
        return [RateLaw(s.target.theta, _mean_field_rate(s))]


class MeanFieldAmplitudeElement(Element):
    """Mean field amplitude element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        K: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        Delta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        omega: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.K = K
        self.Delta = Delta
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the mean field amplitude element."""
        return [
            RateLaw(s.x, (-s.Delta + s.K / 2 * (1 - s.x**2 - s.y**2)) * s.x - s.omega * s.y),
            RateLaw(s.y, (-s.Delta + s.K / 2 * (1 - s.x**2 - s.y**2)) * s.y + s.omega * s.x),
        ]


def _spread_phases(size: int, seed: int) -> list[float]:
    """Return phases drawn uniformly from the circle with a seed."""
    return [float(phase) for phase in np.random.default_rng(seed).uniform(0, 2 * math.pi, size)]


class PhaseCouplingInteraction(PairInteraction):
    """Sine coupling with a phase lag between two phase oscillators, `theta_i' += K sin(theta_j - theta_i - alpha)`.

    Every pair of members closer than the cutoff couples both ways. The members carry the vector state `x`, their
    place, which decides the pairs.

    Args:
        label:
            The label of the interaction.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        **values:
            The values of the declared variables, by name:

            first:
                The path of the group of the first members.

            second:
                The path of the group of the second members.

            cutoff:
                The distance beyond which pairs do not interact.

            K:
                The coupling strength of one pair.

            alpha:
                The phase lag.

    Examples:
        >>> from skcomplex.models.physics.synchronization import PhaseCouplingInteraction
        >>> PhaseCouplingInteraction('coupling', first='phases', second='phases', K=0.1).alpha
        0.0
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        position: str = 'x',
        *,
        K: Annotated[float, ParameterSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.0,
        period: float | None = None,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position)
        self.K = K
        self.alpha = alpha
        self.period = period

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(first, s.K * sin(second - first - s.alpha)),
            RateLaw(second, s.K * sin(first - second - s.alpha)),
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
        PhaseOscillatorElement(f'oscillator_{index}', theta=phases[index], omega=omegas[index])
        for index in range(n_oscillators)
    ]
    interactions = [
        DelayedCouplingInteraction(
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
    return System(elements=elements, interactions=interactions)


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
    oscillators = GroupElement(
        'oscillators',
        element=PhaseOscillatorElement,
        size=n_oscillators,
        values={'theta': _spread_phases(n_oscillators, seed), 'omega': rng.normal(0.0, sigma, n_oscillators)},
    )
    coupling = MeanFieldCouplingInteraction('coupling', target='oscillators', source='oscillators', K=K)
    return System(
        elements=[oscillators],
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
        GroupElement(
            'first',
            element=PhaseOscillatorElement,
            size=n_oscillators,
            values={'theta': rng.normal(0.0, spread, n_oscillators), 'omega': 0.0},
            visual=VisualSpec('phase_ring', {'phase': 'theta'}),
        ),
        GroupElement(
            'second',
            element=PhaseOscillatorElement,
            size=n_oscillators,
            values={'theta': _spread_phases(n_oscillators, seed + 1), 'omega': 0.0},
            visual=VisualSpec('phase_ring', {'phase': 'theta'}),
        ),
    ]
    interactions: list[BaseInteraction] = [
        MeanFieldCouplingInteraction(
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
        elements=populations,
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
    ring = GroupElement(
        'ring',
        element=SitedOscillatorElement,
        size=n_oscillators,
        values={
            'theta': _spread_phases(n_oscillators, seed),
            'omega': omega,
            'x': np.arange(n_oscillators, dtype=float).reshape(-1, 1),
        },
        visual=VisualSpec('phase_ring', {'phase': 'theta', 'position': 'x'}),
    )
    return System(
        elements=[ring],
        interactions=[
            PhaseCouplingInteraction(
                'coupling',
                first='ring',
                second='ring',
                cutoff=R + 0.5,
                period=float(n_oscillators),
                K=weight,
                alpha=alpha,
            ),
            PhaseLagInteraction('lag', target='ring', K=weight, alpha=alpha),
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
    return System(elements=[MeanFieldAmplitudeElement('field', x=r, y=0.0, K=K, Delta=Delta, omega=omega)])
