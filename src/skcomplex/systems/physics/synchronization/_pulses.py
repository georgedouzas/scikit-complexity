"""Pulse coupling systems.

The module covers Winfree oscillators and Mirollo-Strogatz integrate-and-fire oscillators.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction, CrossingInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import maximum, where
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.oscillators import PhaseOscillatorElement, order_parameter


class IntegrateAndFireOscillatorElement(Element):
    """Integrate-and-fire oscillator.

    The oscillator charges as `x' = S - gamma x`, and sends pulses of strength `epsilon`.

    The oscillator charges towards `S / gamma`. Its firing at 1 is a crossing of the group it belongs to, which resets
    the most charged oscillators to 0 and lifts every other one by `epsilon`.

    Args:
        label:
            The label of the oscillator or the group.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial charge.

        S:
            The charging input.

        gamma:
            The leak rate.

        epsilon:
            The strength of a pulse.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import IntegrateAndFireOscillatorElement
        >>> IntegrateAndFireOscillatorElement('oscillator').epsilon
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        S: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.S = S
        self.gamma = gamma
        self.epsilon = epsilon

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the charging of the oscillator.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of the charge.
        """
        s = namespace
        return [RateLaw(s.x, s.S - s.gamma * s.x)]


class WinfreeMeanFieldInteraction(CouplingInteraction):
    """Winfree coupling `theta_i' += K mean_j P(theta_j) R(theta_i)`.

    The mean runs over every member `j` of the group.

    The influence `P(theta) = 1 + cos(theta)` that an oscillator sends and the sensitivity `R(theta) = -sin(theta)` with
    which it responds make every oscillator respond to the mean influence of the group.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of oscillators.

        K:
            The coupling strength.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.synchronization import WinfreeMeanFieldInteraction
        >>> coupling = WinfreeMeanFieldInteraction('coupling', target='oscillators')
        >>> member, other = SimpleNamespace(theta=var('a')), SimpleNamespace(theta=var('b'))
        >>> coupling.function(SimpleNamespace(K=var('K'), member=member, other=other))
        -K*(cos(b) + 1)*sin(a)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            read="theta",
            write="theta",
            function=self._response,
            reduce="mean",
            mode="group",
            include_self=True,
        )
        self.K = K

    def _response(self, namespace: SimpleNamespace) -> Expression:
        """Return the response `K P(theta_j) R(theta_i)` of a member to one other member."""
        return namespace.K * (1 + cos(namespace.other.theta)) * -sin(namespace.member.theta)


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

    Examples:
        >>> from skcomplex.systems.physics.synchronization import WinfreeSystem
        >>> len(WinfreeSystem(5).prepare().system_.interactions)
        1
    """

    field = "physics/synchronization"
    topics = ("synchronization", "oscillators", "phase", "oscillator death")
    references = ("Winfree 1967", "Ariaratnam and Strogatz 2001", "Bick et al. 2020")
    until = 50.0
    dt = 0.5
    observables = tuple(order_parameter("order", "oscillators"))

    def __init__(
        self,
        n_oscillators: int = 50,
        *,
        K: float = 0.5,
        spread: float = 0.1,
        seed: int = 0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_oscillators = n_oscillators
        self.K = K
        self.spread = spread
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        oscillators = GroupElement(
            "oscillators",
            element=PhaseOscillatorElement,
            size=self.n_oscillators,
            values={
                "theta": np.random.default_rng(self.seed).uniform(0, 2 * np.pi, self.n_oscillators),
                "omega": np.linspace(1 - self.spread, 1 + self.spread, self.n_oscillators),
            },
        )
        return System(
            self.label,
            elements=[oscillators],
            interactions=[WinfreeMeanFieldInteraction("coupling", target="oscillators", K=self.K)],
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

    Examples:
        >>> from skcomplex.systems.physics.synchronization import MirolloStrogatzSystem
        >>> MirolloStrogatzSystem(3).prepare().system_.elements[0].prepare().size_
        3
    """

    field = "physics/synchronization"
    topics = ("synchronization", "events", "pulse coupling", "integrate and fire")
    references = ("Mirollo and Strogatz 1990", "Arenas et al. 2008")
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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_oscillators = n_oscillators
        self.S = S
        self.gamma = gamma
        self.epsilon = epsilon
        self.seed = seed

    def _build_system(self) -> System:
        """Return integrate-and-fire oscillators coupled by pulses.

        The oscillators end up firing in unison.
        """
        charges = np.random.default_rng(self.seed).uniform(0.0, 1.0, self.n_oscillators)
        oscillators = GroupElement(
            "oscillators",
            element=IntegrateAndFireOscillatorElement,
            size=self.n_oscillators,
            values={"x": charges, "S": self.S, "gamma": self.gamma, "epsilon": self.epsilon},
        )
        fire = CrossingInteraction(
            "fire",
            target="oscillators",
            guard=lambda s: maximum(s.target.x) - 1,
            reset={
                "x": lambda s: where(
                    s.target.x >= maximum(s.target.x),
                    0,
                    where(s.target.x + s.target.epsilon >= 1, 0, s.target.x + s.target.epsilon),
                ),
            },
            direction=1,
        )
        return System(self.label, elements=[oscillators], interactions=[fire])
