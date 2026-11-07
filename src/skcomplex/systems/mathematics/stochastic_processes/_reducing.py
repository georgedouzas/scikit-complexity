"""Stochastic process systems: sample-space-reducing processes, their cascades, and fragmentation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LifeCycleInteraction, MapInteraction
from ....core.components.system import System
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system


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


@register_system(
    field='mathematics/stochastic_processes',
    tags=('sample space reduction', 'agents', 'zipf', 'power law'),
    references=('Corominas-Murtra, Hanel, and Thurner 2015', 'Thurner et al. 2018'),
    defaults={'until': 100},
)
def sample_space_reducing(n_balls: int = 20, *, n_states: int = 10, lam: float = 1.0) -> System:
    """Return sample-space-reducing processes, whose state visits follow Zipf's law `1 / i` for `lam = 1`.

    Args:
        n_balls:
            The number of independent balls, all started at the top state.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump, the exponent of the visiting distribution.

    Returns:
        system:
            The system, with the group `balls` of state `level`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import sample_space_reducing
        >>> sample_space_reducing(4).elements[0].prepare().size_
        4
    """
    return System(
        elements=[
            GroupElement(
                'balls',
                element=ReducingBallElement,
                size=n_balls,
                values={'level': float(n_states), 'n_states': float(n_states), 'lam': lam},
                variable=True,
            ),
        ],
        interactions=[ReducingJumpInteraction('jump', target='balls')],
    )


@register_system(
    field='mathematics/stochastic_processes',
    tags=('fragmentation', 'agents', 'sample space reduction'),
    references=('Thurner et al. 2018',),
    defaults={'until': 30},
)
def fragmentation(size: int = 1000, *, rate: float = 0.5) -> System:
    """Return the random fragmentation of an object of integer size, whose fragments keep breaking into smaller ones.

    The total size is conserved, and every fragment ends at size 1.

    Args:
        size:
            The size of the initial object.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

    Returns:
        system:
            The system, with the group `fragments` of state `length`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import fragmentation
        >>> fragmentation(50).elements[0].prepare().size_
        1
    """
    return System(
        elements=[
            GroupElement(
                'fragments',
                element=FragmentElement,
                size=1,
                values={'length': float(size), 'rate': rate},
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


@register_system(
    field='mathematics/stochastic_processes',
    tags=('sample space reduction', 'agents', 'cascade', 'power law'),
    references=('Corominas-Murtra, Hanel, and Thurner 2017', 'Thurner et al. 2018'),
    defaults={'until': 100},
)
def cascading_sample_space_reducing(n_cascades: int = 100, *, n_states: int = 100, mu: float = 1.5) -> System:
    """Return cascading sample-space-reducing processes, whose visits follow the power law `i^(-mu)` for any `mu`.

    Every cascade starts with one ball at the top state `n_states`. At every step each ball above state 1 jumps to a
    state drawn uniformly below its own and becomes `mu` balls there, `floor(mu)` and one more with the probability of
    the rest, and a ball at state 1 leaves the cascade. Every ball stays one step, so the states of the balls of all
    frames are the visits, distributed as `i^(-mu)` (Thurner et al. 2018, section 3.3.5.4). For `mu < 1` balls are
    lost on the way, `1 - mu` being the probability of a removal. With a conservation law that makes the states of
    the new balls sum to the state of their parent, the exponent is 2 whatever `mu`, the case of `fragmentation`.

    Args:
        n_cascades:
            The number of independent cascades, run together.

        n_states:
            The number of states.

        mu:
            The multiplication factor, the mean number of balls a ball becomes at every jump.

    Returns:
        system:
            The system, with the group `balls` of state `level`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import cascading_sample_space_reducing
        >>> cascading_sample_space_reducing(4).elements[0].prepare().size_
        4
    """
    return System(
        elements=[
            GroupElement(
                'balls',
                element=CascadingBallElement,
                size=n_cascades,
                values={'level': float(n_states), 'mu': mu},
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
