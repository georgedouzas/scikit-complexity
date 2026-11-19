"""Pulse coupling systems.

The module covers Winfree oscillators and Mirollo-Strogatz integrate-and-fire oscillators.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import EventLaw, RateLaw
from ....core.components.laws.utils import count, maximum, total, where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.oscillators import order_parameter


class IntegrateAndFireOscillatorElement(Element):
    """Integrate-and-fire oscillator with the charge `x`, alone or in a group of `size` members.

    `ChargingInteraction` charges the oscillator and `PulseFiringInteraction` fires it at 1.

    Args:
        label:
            The label of the oscillator or the group.

        x:
            The initial charge, or one charge per member.

        size:
            The number of members of the group, or `None` for one oscillator.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import IntegrateAndFireOscillatorElement
        >>> IntegrateAndFireOscillatorElement('oscillators', x=[0.1, 0.5], size=2).size
        2
    """

    def __init__(self, label: str, *, x: object = 0.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class ChargingInteraction(Interaction):
    """Leaky charging `x' = S - gamma x` towards the level `S / gamma`.

    It charges a capacitor through a resistor, the membrane of a neuron, or the clock of a firefly.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator or the group whose charge `x` rises.

        S:
            The charging input.

        gamma:
            The leak rate.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import ChargingInteraction
        >>> ChargingInteraction('charging', target='oscillators', S=3.0).S
        3.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        S: Annotated[float, RealSet.open(0, oo)] = 2.0,
        gamma: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.S = S
        self.gamma = gamma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the charge of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_ = [RateLaw(x.diff(context.t), context.S - context.gamma * x)]


class PulseFiringInteraction(Interaction):
    """Firing of a group of integrate-and-fire oscillators when the most charged one reaches 1.

    The firing resets the most charged oscillators to 0 and lifts every other one by the pulse `epsilon`. An oscillator
    lifted to 1 fires at the same instant and resets to 0 with the others, as fireflies that flash together or
    pacemaker cells of a heart.

    Args:
        label:
            The label of the interaction, which also names the firings in the trajectory.

        target:
            The path of the group of oscillators.

        epsilon:
            The strength of a pulse.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import PulseFiringInteraction
        >>> PulseFiringInteraction('fire', target='oscillators', epsilon=0.2).epsilon
        0.2
    """

    law_classes = (EventLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        epsilon: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.epsilon = epsilon

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the firing of the group as an event at the crossing of 1 by its largest charge.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, epsilon = context.target.x, context.epsilon
            reset = where(x >= maximum(x), 0, where(x + epsilon >= 1, 0, x + epsilon))
            self.laws_ = [EventLaw(maximum(x) - 1, {x: reset}, direction=1, label=self.label)]


class WinfreeMeanFieldInteraction(Interaction):
    """Winfree dynamics `theta_i' = omega_i + K mean_j P(theta_j) R(theta_i)`.

    The mean runs over every member `j` of the group.

    The phases turn at the natural frequencies `omega` of the target, and the influence `P(theta) = 1 + cos(theta)` that
    an oscillator sends and the sensitivity `R(theta) = -sin(theta)` with which it responds make every oscillator
    respond to the mean influence of the group, as crickets that chirp or cells of a heart that beat.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of oscillators, with the phases `theta` and the natural frequencies `omega`.

        K:
            The coupling strength.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import WinfreeMeanFieldInteraction
        >>> WinfreeMeanFieldInteraction('coupling', target='oscillators', K=2.0).K
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, K: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.K = K

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the phases of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            oscillator = context.target
            theta = oscillator.theta
            response = total(context.K * (1 + cos(oscillator.other.theta)) * -sin(theta)) / count()
            self.laws_ = [RateLaw(theta.diff(context.t), oscillator.omega + response)]


class WinfreeSystem(BaseSystem):
    """The Winfree system.

    The system follows `theta_i' = omega_i + (K / N) sum_j P(theta_j) R(theta_i)`, with the sum over all `j`.

    The influence `P(theta) = 1 + cos(theta)` and the sensitivity `R(theta) = -sin(theta)` make every oscillator respond
    to the mean influence of the population, which every oscillator reads as a group mean. The natural frequencies are
    spread evenly over `[1 - spread, 1 + spread]`. Weak coupling leaves the oscillators incoherent, moderate coupling
    locks them to one frequency, and strong coupling stops them, which is oscillator death. The system has the phases
    `oscillators.theta` and the order parameter `order`.

    Args:
        n_oscillators:
            The number of oscillators.

        K:
            The coupling strength.

        spread:
            The half width of the interval of natural frequencies around 1.

        seed:
            The seed of the initial phases.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import WinfreeSystem
        >>> len(WinfreeSystem(5).prepare().system_.interactions)
        1
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'oscillators', 'phase', 'oscillator death')
    references = ('Winfree 1967', 'Ariaratnam and Strogatz 2001', 'Bick et al. 2020')
    until = 50.0
    dt = 0.5
    observables = tuple(order_parameter('order', 'oscillators'))

    def __init__(
        self,
        n_oscillators: int = 50,
        *,
        K: float = 0.5,
        spread: float = 0.1,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_oscillators = n_oscillators
        self.K = K
        self.spread = spread
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Winfree oscillators.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        values = {
            'theta': np.random.default_rng(self.seed).uniform(0, 2 * np.pi, self.n_oscillators),
            'omega': np.linspace(1 - self.spread, 1 + self.spread, self.n_oscillators),
        }
        self.system_ = System(
            self.label,
            elements=[Element('oscillators', values=values, size=self.n_oscillators)],
            interactions=[WinfreeMeanFieldInteraction('coupling', target='oscillators', K=self.K)],
        )


class MirolloStrogatzSystem(BaseSystem):
    """Integrate-and-fire oscillators coupled by pulses.

    The oscillators end up firing in unison.

    Every oscillator of the group charges as `x' = S - gamma x`, a concave rise, and fires when it reaches 1. Firing is
    an event whose guard is the group maximum of the charges, and it resets the most charged oscillators to 0 and lifts
    every other one by `epsilon`. An oscillator lifted to 1 fires at the same instant and is absorbed into the group
    that fired, so the groups merge until all oscillators fire together. The system has the charges `oscillators.x` of
    the group members and the firings as `fire` events.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import MirolloStrogatzSystem
        >>> MirolloStrogatzSystem(3).prepare().system_.elements[0].size
        3
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'events', 'pulse coupling', 'integrate and fire')
    references = ('Mirollo and Strogatz 1990', 'Arenas et al. 2008')
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        n_oscillators: int = 5,
        *,
        S: float = 2.0,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_oscillators = n_oscillators
        self.S = S
        self.gamma = gamma
        self.epsilon = epsilon
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of integrate-and-fire oscillators coupled by pulses.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        charges = np.random.default_rng(self.seed).uniform(0.0, 1.0, self.n_oscillators)
        self.system_ = System(
            self.label,
            elements=[IntegrateAndFireOscillatorElement('oscillators', x=charges, size=self.n_oscillators)],
            interactions=[
                ChargingInteraction('charging', target='oscillators', S=self.S, gamma=self.gamma),
                PulseFiringInteraction('fire', target='oscillators', epsilon=self.epsilon),
            ],
        )
