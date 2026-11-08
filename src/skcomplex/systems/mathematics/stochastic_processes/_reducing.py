"""Stochastic process systems: sample-space-reducing processes, their cascades, and fragmentation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LifeCycleInteraction, MapInteraction
from ....core.components.system import BaseSystem, System
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class CascadingBallView(AgentView):
    """View of a ball of a cascading sample-space-reducing process, with its states and parameters as attributes."""

    level: float
    mu: float


class CascadingBallElement(Element):
    """Ball of a cascading sample-space-reducing process, which multiplies into `mu` balls at every state it hits.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial state.

        mu:
            The mean number of balls a ball becomes at every jump, the multiplication factor.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import CascadingBallElement
        >>> group = GroupElement('balls', element=CascadingBallElement, size=5, values={'mu': 2.0})
        >>> group.label, group.element.__name__, group.size
        ('balls', 'CascadingBallElement', 5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'level'}),
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 100.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.mu = mu

    @classmethod
    def _get_view(cls) -> type[CascadingBallView]:
        """Return the view of the agents, which types their states and parameters."""
        return CascadingBallView

    @staticmethod
    def stops(agent: CascadingBallView, system: SystemView) -> bool:
        """Return whether a ball has reached the ground state 1, where its cascade ends.

        Args:
            agent:
                The ball.

            system:
                The system.

        Returns:
            stopped:
                Whether the ball is at state 1.
        """
        return agent.level <= 1

    @staticmethod
    def multiplies(agent: CascadingBallView, system: SystemView) -> int:
        """Return the number of balls a ball becomes, `floor(mu)` and one more with the probability of the rest.

        Args:
            agent:
                The ball.

            system:
                The system.

        Returns:
            count:
                The number of new balls, of mean `mu`.
        """
        whole = math.floor(agent.mu)
        return whole + int(system.rng.random() < agent.mu - whole)

    @staticmethod
    def jumps(agent: CascadingBallView, system: SystemView) -> dict[str, float]:
        """Return the state of a new ball, drawn uniformly below the state of its parent.

        Args:
            agent:
                The parent ball.

            system:
                The system.

        Returns:
            values:
                The state of the new ball.
        """
        return {'level': float(system.rng.integers(1, round(agent.level)))}


class FragmentView(AgentView):
    """View of a fragment, with its states and parameters as attributes."""

    length: float
    rate: float


class FragmentElement(Element):
    """Fragment of integer size, which the breaking of the system cuts at a uniformly random point into two fragments.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        length:
            The initial size.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import FragmentElement
        >>> group = GroupElement('fragments', element=FragmentElement, size=1, values={'length': 100.0})
        >>> group.label, group.element.__name__, group.size
        ('fragments', 'FragmentElement', 1)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'length'}),
        *,
        length: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1000.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.length = length
        self.rate = rate

    @classmethod
    def _get_view(cls) -> type[FragmentView]:
        """Return the view of the agents, which types their states and parameters."""
        return FragmentView

    @staticmethod
    def breaks(agent: FragmentView, system: SystemView) -> float:
        """Return the probability that a fragment breaks in a step, zero at size 1.

        Args:
            agent:
                The fragment.

            system:
                The system.

        Returns:
            probability:
                The breaking probability, or zero for a fragment of size 1.
        """
        return agent.rate if round(agent.length) > 1 else 0.0

    @staticmethod
    def cut(agent: FragmentView, system: SystemView) -> float:
        """Return the share of a breaking fragment that stays with it, cut at a uniformly random whole point.

        Args:
            agent:
                The fragment.

            system:
                The system.

        Returns:
            share:
                The share of the size left of the cut.
        """
        length = round(agent.length)
        return int(system.rng.integers(1, length)) / length


class ReducingBallView(AgentView):
    """View of a ball of a sample-space-reducing process, with its states and parameters as attributes."""

    level: float
    n_states: float
    lam: float


class ReducingBallElement(Element):
    """Ball of the sample-space-reducing process, whose state the reducing jump moves to a random lower state.

    With probability `lam` the jump takes the ball uniformly to a state below its own, and otherwise uniformly to any
    state, and the ball restarts from the top state after it reaches state 1.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial state, from 1 to `n_states`.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import ReducingBallElement
        >>> group = GroupElement('balls', element=ReducingBallElement, size=3, values={'n_states': 10.0})
        >>> group.label, group.element.__name__, group.size
        ('balls', 'ReducingBallElement', 3)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'level'}),
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 10.0,
        n_states: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.n_states = n_states
        self.lam = lam

    @classmethod
    def _get_view(cls) -> type[ReducingBallView]:
        """Return the view of the agents, which types their states and parameters."""
        return ReducingBallView


class ReducingJumpInteraction(MapInteraction):
    """Jump of the sample-space-reducing process, to a random lower state, to any state, or back to the top state.

    A ball at state 1 restarts from the top state `n_states`. Otherwise it jumps with probability `lam` uniformly to a
    state below its own, and else uniformly to any state, so the states are visited with probabilities proportional to
    `i^(-lam)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of balls.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ReducingJumpInteraction
        >>> ReducingJumpInteraction('jump', target='balls').state
        'level'
    """

    draws = True

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, state='level', function=self._get_level, form='agent')

    def _get_level(self, agent: ReducingBallView, system: SystemViewProtocol) -> float:
        """Return the next state of a ball."""
        top = round(agent.n_states)
        if agent.level <= 1:
            return float(top)
        if system.rng.random() < agent.lam:
            return float(system.rng.integers(1, round(agent.level)))
        return float(system.rng.integers(1, top + 1))


class SampleSpaceReducingSystem(BaseSystem):
    """Sample-space-reducing processes, whose state visits follow Zipf's law `1 / i` for `lam = 1`.

    The system has the group `balls` of state `level`.

    Args:
        n_balls:
            The number of independent balls, all started at the top state.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump, the exponent of the visiting distribution.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import SampleSpaceReducingSystem
        >>> SampleSpaceReducingSystem(4).prepare().system_.elements[0].prepare().size_
        4
    """

    field = 'mathematics/stochastic_processes'
    topics = ('sample space reduction', 'agents', 'zipf', 'power law')
    references = ('Corominas-Murtra, Hanel, and Thurner 2015', 'Thurner et al. 2018')
    until = 100

    def __init__(
        self,
        n_balls: int = 20,
        *,
        n_states: int = 10,
        lam: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_balls = n_balls
        self.n_states = n_states
        self.lam = lam

    def _build_system(self) -> System:
        """Return sample-space-reducing processes, whose state visits follow Zipf's law `1 / i` for `lam = 1`."""
        return System(
            self.label,
            elements=[
                GroupElement(
                    'balls',
                    element=ReducingBallElement,
                    size=self.n_balls,
                    values={'level': float(self.n_states), 'n_states': float(self.n_states), 'lam': self.lam},
                    variable=True,
                ),
            ],
            interactions=[ReducingJumpInteraction('jump', target='balls')],
        )


class FragmentationSystem(BaseSystem):
    """The random fragmentation of an object of integer size, whose fragments keep breaking into smaller ones.

    The total size is conserved, and every fragment ends at size 1. The system has the group `fragments` of state
    `length`.

    Args:
        size:
            The size of the initial object.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import FragmentationSystem
        >>> FragmentationSystem(50).prepare().system_.elements[0].prepare().size_
        1
    """

    field = 'mathematics/stochastic_processes'
    topics = ('fragmentation', 'agents', 'sample space reduction')
    references = ('Thurner et al. 2018',)
    until = 30

    def __init__(
        self,
        size: int = 1000,
        *,
        rate: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.rate = rate

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[
                GroupElement(
                    'fragments',
                    element=FragmentElement,
                    size=1,
                    values={'length': float(self.size), 'rate': self.rate},
                    variable=True,
                ),
            ],
            interactions=[
                LifeCycleInteraction(
                    'breaking',
                    target='fragments',
                    birth=FragmentElement.breaks,
                    split=('length',),
                    keep=FragmentElement.cut,
                ),
            ],
        )


class CascadingSampleSpaceReducingSystem(BaseSystem):
    """Cascading sample-space-reducing processes, whose visits follow the power law `i^(-mu)` for any `mu`.

    Every cascade starts with one ball at the top state `n_states`. At every step each ball above state 1 jumps to a
    state drawn uniformly below its own and becomes `mu` balls there, `floor(mu)` and one more with the probability of
    the rest, and a ball at state 1 leaves the cascade. Every ball stays one step, so the states of the balls of all
    frames are the visits, distributed as `i^(-mu)` (Thurner et al. 2018, section 3.3.5.4). For `mu < 1` balls are lost
    on the way, `1 - mu` being the probability of a removal. With a conservation law that makes the states of the new
    balls sum to the state of their parent, the exponent is 2 whatever `mu`, the case of `FragmentationSystem`. The
    system has the group `balls` of state `level`.

    Args:
        n_cascades:
            The number of independent cascades, run together.

        n_states:
            The number of states.

        mu:
            The multiplication factor, the mean number of balls a ball becomes at every jump.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import CascadingSampleSpaceReducingSystem
        >>> CascadingSampleSpaceReducingSystem(4).prepare().system_.elements[0].prepare().size_
        4
    """

    field = 'mathematics/stochastic_processes'
    topics = ('sample space reduction', 'agents', 'cascade', 'power law')
    references = ('Corominas-Murtra, Hanel, and Thurner 2017', 'Thurner et al. 2018')
    until = 100

    def __init__(
        self,
        n_cascades: int = 100,
        *,
        n_states: int = 100,
        mu: float = 1.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_cascades = n_cascades
        self.n_states = n_states
        self.mu = mu

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[
                GroupElement(
                    'balls',
                    element=CascadingBallElement,
                    size=self.n_cascades,
                    values={'level': float(self.n_states), 'mu': self.mu},
                    variable=True,
                ),
            ],
            interactions=[
                LifeCycleInteraction(
                    'cascade',
                    target='balls',
                    death=CascadingBallElement.stops,
                    birth=1.0,
                    offspring=CascadingBallElement.multiplies,
                    inherit=('mu',),
                    values=CascadingBallElement.jumps,
                    replace=True,
                ),
            ],
        )
