"""Pulse coupling models: Winfree oscillators and Mirollo-Strogatz integrate-and-fire oscillators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from sage.all__sagemath_symbolics import cos, sin

from ....core.base import Domain, Expression, parameter, state
from ....core.model import (
    Element,
    EuclideanSpace,
    Event,
    Interaction,
    Law,
    PairInteraction,
    Rate,
    Symbols,
    System,
    where,
)
from ....core.runtime import register_model
from ._utils import _order_parameter


def _influence(theta: Expression) -> Expression:
    """Return the Winfree influence `1 + cos(theta)` that an oscillator sends."""
    return 1 + cos(theta)


def _sensitivity(theta: Expression) -> Expression:
    """Return the Winfree sensitivity `-sin(theta)` with which an oscillator responds."""
    return -sin(theta)


def _integrate_and_fire_laws(s: Symbols) -> list[Law]:
    """Return the charging of every oscillator and its firing, which resets it and kicks the others up by epsilon."""
    levels = list(s.x)
    laws: list[Law] = [Rate(level, s.S - s.gamma * level) for level in levels]
    for index, level in enumerate(levels):
        reset = {
            other: 0 if position == index else where(other + s.epsilon >= 1, 0, other + s.epsilon)
            for position, other in enumerate(levels)
        }
        laws.append(Event(level - 1, reset, direction=1, label=f'fire {index}'))
    return laws


WinfreeOscillator = Element.define(
    'WinfreeOscillator',
    states={'theta': 0.0, 'x': state(default=0.0, vector=True)},
    parameters={'omega': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)

WinfreeSelfInfluence = Interaction.define(
    'WinfreeSelfInfluence',
    roles=['target'],
    parameters={'K': 1.0},
    laws=lambda s: [Rate(s.target.theta, s.K * _influence(s.target.theta) * _sensitivity(s.target.theta))],
)

IntegrateAndFireOscillators = Element.define(
    'IntegrateAndFireOscillators',
    states={'x': state(default=0.0, vector=True)},
    parameters={
        'S': parameter(default=2.0, domain=Domain.POSITIVE),
        'gamma': parameter(default=1.0, domain=Domain.POSITIVE),
        'epsilon': parameter(default=0.1, domain=Domain.NONNEGATIVE),
    },
    laws=_integrate_and_fire_laws,
)


class WinfreeCoupling(PairInteraction):
    """Winfree coupling, where every oscillator responds to the influence of the others, `K P(theta_j) R(theta_i)`.

    The influence is `P(theta) = 1 + cos(theta)` and the sensitivity `R(theta) = -sin(theta)`. Every pair of members
    closer than the cutoff couples both ways.

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

    Examples:
        >>> from skcomplex.models.physics.synchronization import WinfreeCoupling
        >>> WinfreeCoupling('coupling', first='phases', second='phases', K=0.1).K
        0.1
    """

    K: float = parameter(default=1.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the response of both members of a pair to the influence of the other.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        first, second = s.first.theta, s.second.theta
        return [
            Rate(first, s.K * _influence(second) * _sensitivity(first)),
            Rate(second, s.K * _influence(first) * _sensitivity(second)),
        ]


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'phase', 'oscillator death'),
    references=('Winfree 1967', 'Ariaratnam and Strogatz 2001', 'Bick et al. 2020'),
    defaults={'until': 50.0, 'dt': 0.5},
)
def winfree(n_oscillators: int = 50, *, K: float = 0.5, spread: float = 0.1, seed: int = 0) -> System:
    """Return the Winfree model, `theta_i' = omega_i + (K / N) sum_j P(theta_j) R(theta_i)`, with the sum over all `j`.

    The influence `P(theta) = 1 + cos(theta)` and the sensitivity `R(theta) = -sin(theta)` make every oscillator respond
    to the mean influence of the population. The natural frequencies are spread evenly over `[1 - spread, 1 + spread]`.
    Weak coupling leaves the oscillators incoherent, moderate coupling locks them to one frequency, and strong coupling
    stops them, which is oscillator death.

    Args:
        n_oscillators:
            The number of oscillators.

        K:
            The coupling strength.

        spread:
            The half width of the interval of natural frequencies around 1.

        seed:
            The seed of the initial phases.

    Returns:
        system:
            The model, with the phases `oscillators.theta` and the order parameter `order`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import winfree
        >>> len(winfree(5).interactions)
        2
    """
    oscillators = WinfreeOscillator.group(
        'oscillators',
        n_oscillators,
        theta=np.random.default_rng(seed).uniform(0, 2 * np.pi, n_oscillators),
        omega=np.linspace(1 - spread, 1 + spread, n_oscillators),
        x=np.zeros((n_oscillators, 1)),
    )
    weight = K / n_oscillators
    return System(
        elements=[oscillators],
        interactions=[
            WinfreeCoupling('coupling', first='oscillators', second='oscillators', K=weight),
            WinfreeSelfInfluence('self_influence', target='oscillators', K=weight),
        ],
        observables=_order_parameter('order', 'oscillators'),
        space=EuclideanSpace(1),
    )


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'events', 'pulse coupling', 'integrate and fire'),
    references=('Mirollo and Strogatz 1990', 'Arenas et al. 2008'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def mirollo_strogatz(
    n_oscillators: int = 5,
    *,
    S: float = 2.0,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 0,
) -> System:
    """Return integrate-and-fire oscillators coupled by pulses, which end up firing in unison.

    Every oscillator charges as `x' = S - gamma x`, a concave rise, and fires when it reaches 1. Firing is an event
    that resets it to 0 and lifts every other oscillator by `epsilon`. An oscillator lifted to 1 fires at the same
    instant and is absorbed into the group that fired, so the groups merge until all oscillators fire together.

    Args:
        n_oscillators:
            The number of oscillators.

        S:
            The charging input.

        gamma:
            The leak rate, below `S` so that the oscillators fire.

        epsilon:
            The strength of a pulse.

        seed:
            The seed of the initial charges.

    Returns:
        system:
            The model, with the charges `oscillators.x`, one component per oscillator.

    Examples:
        >>> from skcomplex.models.physics.synchronization import mirollo_strogatz
        >>> len(mirollo_strogatz(3).elements[0].x)
        3
    """
    charges = np.random.default_rng(seed).uniform(0.0, 1.0, n_oscillators)
    return System(
        elements=[IntegrateAndFireOscillators('oscillators', x=list(charges), S=S, gamma=gamma, epsilon=epsilon)],
        space=EuclideanSpace(n_oscillators),
    )
