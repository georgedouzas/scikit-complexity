"""Willing majority systems: agents who adopt the majority of a random group.

The majority is that of opinions. They adopt it with their willingness.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import NN
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role


class WillingVoterView(AgentViewProtocol, Protocol):
    """View of a voter of the majority rule with willingness.

    Its states are attributes.
    """

    opinion: float
    willingness: float


class WillingVoterElement(Element):
    """Voter with a binary opinion and a willingness to follow the majority.

    It is a zealot at willingness 0.

    Args:
        label:
            The label of the voter or the group.

        opinion:
            The opinion, 0 for `B` or 1 for `A`.

        willingness:
            The probability that the voter adopts the majority opinion of the group it consults.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import WillingVoterElement
        >>> WillingVoterElement('voter').willingness
        1.0
    """

    visual = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, NN] = 0.0,
        willingness: Annotated[object, RealSet.closed(0, 1)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.opinion = opinion
        self.willingness = willingness


class WillingMajorityInteraction(Interaction):
    """Majority rule with willingness.

    A random agent adopts the majority opinion of a random group. Every step has one update per agent. An update chooses
    an agent uniformly, then an odd number `group_size` of other agents uniformly, and the agent adopts the majority
    opinion of the group with the probability of its willingness.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of voters.

        group_size:
            The odd number of other agents an agent consults.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import WillingMajorityInteraction
        >>> WillingMajorityInteraction('majority', target='voters').group_size
        3
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        group_size: Annotated[int, NN] = 3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.group_size = group_size

    def _prepare(self, s: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the majority rule."""
        super()._prepare(s)
        if isinstance(s, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(s, self.follow)]

    def follow(self, system: SystemView) -> None:
        """Update as many random agents as there are agents.

        Each agent follows the majority of a random group.

        Args:
            system:
                The system.
        """
        voters = cast(list[WillingVoterView], system.get_agents(self.target))
        size = len(voters)
        group_size = int(self.group_size)
        if size <= group_size:
            return
        opinions = np.array([voter.opinion for voter in voters])
        willingness = np.array([voter.willingness for voter in voters])
        rng = system.rng
        for chosen in rng.integers(size, size=size).tolist():
            others = rng.choice(size - 1, group_size, replace=False)
            others = others + (others >= chosen)
            if rng.random() < willingness[chosen]:
                opinions[chosen] = float(2 * opinions[others].sum() > group_size)
        for voter, opinion in zip(voters, opinions.tolist(), strict=True):
            voter.opinion = opinion


class WillingMajoritySystem(BaseSystem):
    """The majority rule with willingness.

    Zealots of willingness 0 never change their opinion. Every agent holds opinion `A` or `B`, drawn with probability
    1/2, and a willingness `w_i`. An update chooses an agent and an odd number `n` of other agents at random, and the
    agent adopts the majority opinion of the group with the probability `w_i` (Jensen 2023 section 11.5, Exercise 4).
    Every agent has `w_i = willingness`, except a fraction `zealots` of them with `w_i = 0`. With every agent willing,
    the population reaches consensus. Zealots of both opinions prevent it, while a small group of zealots of one
    opinion, `zealot_opinion`, wins over everyone. The system has the group `voters`.

    Args:
        n_agents:
            The number of agents.

        group_size:
            The odd number `n` of other agents an agent consults.

        zealots:
            The fraction `f` of agents with willingness 0.

        zealot_opinion:
            The opinion, 1 for `A` or 0 for `B`, every zealot starts with, or `None` for random opinions.

        willingness:
            The willingness of the other agents.

        random_state:
            The seed or generator of the opinions and of the zealots.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import WillingMajoritySystem
        >>> WillingMajoritySystem(30).prepare().system_.elements[0].size
        30
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'majority', 'zealots', 'consensus')
    references = ('Jensen 2023',)
    until = 50

    def __init__(
        self,
        n_agents: int = 200,
        *,
        group_size: int = 3,
        zealots: float = 0.0,
        zealot_opinion: float | None = None,
        willingness: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.group_size = group_size
        self.zealots = zealots
        self.zealot_opinion = zealot_opinion
        self.willingness = willingness
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        opinions = rng.integers(2, size=self.n_agents).astype(float)
        stubborn = rng.random(self.n_agents) < self.zealots
        if self.zealot_opinion is not None:
            opinions[stubborn] = self.zealot_opinion
        voters = WillingVoterElement(
            'voters',
            opinion=opinions,
            willingness=np.where(stubborn, 0.0, self.willingness),
            size=self.n_agents,
        )
        majority = WillingMajorityInteraction('majority', target='voters', group_size=self.group_size)
        self.system_ = System(self.label, elements=[voters], interactions=[majority])
