"""Pulse coupling models: Winfree oscillators and Mirollo-Strogatz integrate-and-fire oscillators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from sage.all__sagemath_symbolics import cos, sin

from ....core.base import Domain, Expression, parameter
from ....core.model import Element, Event, Interaction, Law, Rate, Symbols, System, maximum, mean, where
from ....core.runtime import register_model
from ._utils import _order_parameter


def _influence(theta: Expression) -> Expression:
    """Return the Winfree influence `1 + cos(theta)` that an oscillator sends."""
    return 1 + cos(theta)


def _sensitivity(theta: Expression) -> Expression:
    """Return the Winfree sensitivity `-sin(theta)` with which an oscillator responds."""
    return -sin(theta)


def _integrate_and_fire_laws(s: Symbols) -> list[Law]:
    """Return the charging of an oscillator and the firing of the most charged ones, which kick the others up."""
    largest = maximum(s.x)
    kicked = where(s.x + s.epsilon >= 1, 0, s.x + s.epsilon)
    return [
        Rate(s.x, s.S - s.gamma * s.x),
        Event(largest - 1, {s.x: where(s.x >= largest, 0, kicked)}, direction=1, label='fire'),
    ]


WinfreeOscillator = Element.define(
    'WinfreeOscillator',
    states={'theta': 0.0},
    parameters={'omega': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)

WinfreeMeanField = Interaction.define(
    'WinfreeMeanField',
    roles=['target'],
    parameters={'K': 1.0},
    laws=lambda s: [
        Rate(s.target.theta, s.K * mean(_influence(s.target.theta)) * _sensitivity(s.target.theta)),
    ],
)

IntegrateAndFireOscillator = Element.define(
    'IntegrateAndFireOscillator',
    states={'x': 0.0},
    parameters={
        'S': parameter(default=2.0, domain=Domain.POSITIVE),
        'gamma': parameter(default=1.0, domain=Domain.POSITIVE),
        'epsilon': parameter(default=0.1, domain=Domain.NONNEGATIVE),
    },
    laws=_integrate_and_fire_laws,
)


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'phase', 'oscillator death'),
    references=('Winfree 1967', 'Ariaratnam and Strogatz 2001', 'Bick et al. 2020'),
    defaults={'until': 50.0, 'dt': 0.5},
)
def winfree(n_oscillators: int = 50, *, K: float = 0.5, spread: float = 0.1, seed: int = 0) -> System:
    """Return the Winfree model, `theta_i' = omega_i + (K / N) sum_j P(theta_j) R(theta_i)`, with the sum over all `j`.

    The influence `P(theta) = 1 + cos(theta)` and the sensitivity `R(theta) = -sin(theta)` make every oscillator respond
    to the mean influence of the population, which every oscillator reads as a group mean. The natural frequencies are
    spread evenly over `[1 - spread, 1 + spread]`. Weak coupling leaves the oscillators incoherent, moderate coupling
    locks them to one frequency, and strong coupling stops them, which is oscillator death.

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
        1
    """
    oscillators = WinfreeOscillator.group(
        'oscillators',
        n_oscillators,
        theta=np.random.default_rng(seed).uniform(0, 2 * np.pi, n_oscillators),
        omega=np.linspace(1 - spread, 1 + spread, n_oscillators),
    )
    return System(
        elements=[oscillators],
        interactions=[WinfreeMeanField('coupling', target='oscillators', K=K)],
        observables=_order_parameter('order', 'oscillators'),
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

    Every oscillator of the group charges as `x' = S - gamma x`, a concave rise, and fires when it reaches 1. Firing is
    an event whose guard is the group maximum of the charges, and it resets the most charged oscillators to 0 and lifts
    every other one by `epsilon`. An oscillator lifted to 1 fires at the same instant and is absorbed into the group
    that fired, so the groups merge until all oscillators fire together.

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
            The model, with the charges `oscillators.x` of the group members and the firings as `fire` events.

    Examples:
        >>> from skcomplex.models.physics.synchronization import mirollo_strogatz
        >>> mirollo_strogatz(3).elements[0].size
        3
    """
    charges = np.random.default_rng(seed).uniform(0.0, 1.0, n_oscillators)
    oscillators = IntegrateAndFireOscillator.group(
        'oscillators',
        n_oscillators,
        x=charges,
        S=S,
        gamma=gamma,
        epsilon=epsilon,
    )
    return System(elements=[oscillators])
