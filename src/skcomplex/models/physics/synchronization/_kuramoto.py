"""Kuramoto models: phase oscillators coupled all to all, in populations, on rings, with delays, and as a mean field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.elements import GroupElement
from ....core.interactions import BaseInteraction, Interaction, RateInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import delayed, mean
from ....core.spaces import ContinuousSpace, RadiusNeighborhood
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, VisualSpec
from ..._catalog import register_model
from ..common.oscillators import (
    KuramotoCouplingInteraction,
    PhaseOscillatorElement,
    SitedOscillatorElement,
    StuartLandauElement,
)
from ._utils import _order_parameter


class DelayedCouplingInteraction(RateInteraction):
    """Delayed sine coupling of one phase oscillator to another, `theta_t' += K sin(theta_s(t - tau) - theta_t)`.

    The target reads the phase of the source as it was a delay `tau` earlier.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        source:
            The path of the oscillator whose delayed phase the coupling reads.

        target:
            The path of the oscillator whose phase the coupling turns.

        K:
            The coupling strength.

        tau:
            The delay.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.models.physics.synchronization import DelayedCouplingInteraction
        >>> coupling = DelayedCouplingInteraction('coupling', source='a', target='b').check().prepare()
        >>> namespace = SimpleNamespace(source=SimpleNamespace(theta=var('a')), target=SimpleNamespace(theta=var('b')))
        >>> [law.value for law in coupling.run(SimpleNamespace(**vars(namespace), K=var('K'), tau=1)).laws_]
        [K*sin(-b + delayed(a, 1))]
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=target, state='theta')
        self.source = source
        self.K = K
        self.tau = tau

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return `K sin(theta_s(t - tau) - theta_t)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The rate of the phase of the target.
        """
        return namespace.K * sin(delayed(namespace.source.theta, namespace.tau) - namespace.target.theta)


class MeanFieldCouplingInteraction(Interaction):
    """Kuramoto-Sakaguchi coupling of one population to the mean field of another, possibly itself.

    Every member `i` of the target turns by `K mean_j sin(theta_j - theta_i - alpha)` over every member `j` of the
    source, which it reads from the means of `cos theta_j` and `sin theta_j`, so the coupling costs time linear in the
    sizes of the populations.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the population whose phases turn.

        source:
            The path of the population whose mean field the target reads.

        K:
            The coupling strength.

        alpha:
            The phase lag.

    Examples:
        >>> from skcomplex.models.physics.synchronization import MeanFieldCouplingInteraction
        >>> MeanFieldCouplingInteraction('coupling', target='first', source='second', alpha=0.1).alpha
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.K = K
        self.alpha = alpha

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate `K (<sin theta_j> cos(theta_i + alpha) - <cos theta_j> sin(theta_i + alpha))`.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of the phase of the target.
        """
        shifted = namespace.target.theta + namespace.alpha
        sines, cosines = mean(sin(namespace.source.theta)), mean(cos(namespace.source.theta))
        return [RateLaw(namespace.target.theta, namespace.K * (sines * cos(shifted) - cosines * sin(shifted)))]


def _spread_phases(size: int, seed: int) -> list[float]:
    """Return phases drawn uniformly from the circle with a seed."""
    return [float(phase) for phase in np.random.default_rng(seed).uniform(0, 2 * math.pi, size)]


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
    coupling = KuramotoCouplingInteraction(
        'coupling',
        target='oscillators',
        K=K,
        mode='group',
        reduce='mean',
        include_self=True,
    )
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
        >>> nonlocal_ring_chimera(20, R=3).interactions[0].neighborhood.radius
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
            KuramotoCouplingInteraction(
                'coupling',
                target='ring',
                K=weight,
                alpha=alpha,
                include_self=True,
                neighborhood=RadiusNeighborhood(R + 0.5),
            ),
        ],
        space=ContinuousSpace((float(n_oscillators),), boundary='periodic'),
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
    parameter `z = x + i y` follows exactly `z' = (i omega - Delta) z + (K / 2)(1 - |z|²) z`, a Stuart-Landau
    oscillator with the growth rate `K / 2 - Delta` and the saturation `K / 2`. Its modulus settles at
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
        >>> ott_antonsen(K=3.0).elements[0].kappa
        1.5
    """
    return System(elements=[StuartLandauElement('field', x=r, y=0.0, mu=K / 2 - Delta, omega=omega, kappa=K / 2)])
