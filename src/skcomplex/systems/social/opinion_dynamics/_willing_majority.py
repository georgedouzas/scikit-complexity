"""Willing majority systems: agents who adopt, with their willingness, the majority opinion of a random group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.system import System
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ._willing_voter_view import WillingVoterView


class WillingVoterElement(Element):
    """Voter with a binary opinion and a willingness to follow the majority, a zealot at willingness 0.

    Args:
        label:
            The label of the voter or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The opinion, 0 for `B` or 1 for `A`.

        willingness:
            The probability that the voter adopts the majority opinion of the group it consults.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import WillingVoterElement
        >>> WillingVoterElement('voter').willingness
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
        willingness: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.willingness = willingness

    @classmethod
    def _get_view(cls) -> type[WillingVoterView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return WillingVoterView


class WillingMajorityInteraction(Interaction):
    """Majority rule with willingness, where a random agent adopts the majority opinion of a random group.

    Every step has one update per agent. An update chooses an agent uniformly, then an odd number `group_size` of
    other agents uniformly, and the agent adopts the majority opinion of the group with the probability of its
    willingness.

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

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        group_size: Annotated[int, ParameterSpec(domain=NN)] = 3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.group_size = group_size

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the majority rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.follow)]

    def follow(self, system: SystemView) -> None:
        """Update as many random agents as there are agents, each by the majority of a random group.

        Args:
            system:
                The system.
        """
        voters = system.agents(self.target, WillingVoterView)
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


@register_system(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'majority', 'zealots', 'consensus'),
    references=('Jensen 2023',),
    defaults={'until': 50},
)
def willing_majority(
    n_agents: int = 200,
    *,
    group_size: int = 3,
    zealots: float = 0.0,
    zealot_opinion: float | None = None,
    willingness: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the majority rule with willingness, where zealots of willingness 0 never change their opinion.

    Every agent holds opinion `A` or `B`, drawn with probability 1/2, and a willingness `w_i`. An update chooses an
    agent and an odd number `n` of other agents at random, and the agent adopts the majority opinion of the group with
    the probability `w_i` (Jensen 2023 section 11.5, Exercise 4). Every agent has `w_i = willingness`, except a
    fraction `zealots` of them with `w_i = 0`. With every agent willing, the population reaches consensus. Zealots of
    both opinions prevent it, while a small group of zealots of one opinion, `zealot_opinion`, wins over everyone.

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

    Returns:
        system:
            The system, with the group `voters`.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import willing_majority
        >>> willing_majority(30).elements[0].prepare().size_
        30
    """
    rng = np.random.default_rng(random_state)
    opinions = (rng.random(n_agents) < 0.5).astype(float)
    stubborn = rng.random(n_agents) < zealots
    if zealot_opinion is not None:
        opinions[stubborn] = zealot_opinion
    voters = GroupElement(
        'voters',
        element=WillingVoterElement,
        size=n_agents,
        values={'opinion': opinions, 'willingness': np.where(stubborn, 0.0, willingness)},
    )
    majority = WillingMajorityInteraction('majority', target='voters', group_size=group_size)
    return System(elements=[voters], interactions=[majority])
