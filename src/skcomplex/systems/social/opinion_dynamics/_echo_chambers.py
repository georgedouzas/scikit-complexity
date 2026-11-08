"""Echo chamber systems: radicalization of opinions over a network.

The network is activity-driven. The network has homophily.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class RadicalView(AgentView):
    """View of an agent of the echo chamber system, with its states as attributes."""

    opinion: float
    activity: float


PAIR = 2


class RadicalElement(Element):
    """Agent with an unbounded opinion and an activity.

    The sign of the opinion is its stance and its size is its conviction.

    Args:
        label:
            The label of the agent or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The opinion `x`.

        activity:
            The probability `a` that the agent contacts others in an integration step.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import RadicalElement
        >>> RadicalElement('agent', activity=0.1).activity
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (-3.0, 3.0)}),
        *,
        opinion: Annotated[float, StateSpec()] = 0.0,
        activity: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.activity = activity

    @classmethod
    def _get_view(cls) -> type[RadicalView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return RadicalView


class RadicalizationInteraction(Interaction):
    """Opinion dynamics of Baumann and coauthors.

    The opinions follow `dx_i/dt = -x_i + K sum_j A_ij(t) tanh(alpha x_j)`. Every step of the system integrates the
    opinions with `substeps` Euler steps of length `dt`, and draws a new temporal network for every one of them. In a
    draw, every agent is active with the probability of its activity and contacts `contacts` distinct others, each
    chosen with the probability `|x_i - x_j|^-beta` normalized over the others. A contacted agent receives the influence
    of the agent that contacts it, which receives the influence back when the contact is reciprocated, with the
    probability `reciprocity`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents.

        strength:
            The social interaction strength `K`.

        controversy:
            The controversialness `alpha` of the issue, the steepness of the influence `tanh(alpha x)`.

        homophily:
            The exponent `beta` of the homophily of the contacts.

        contacts:
            The number `m` of distinct agents an active agent contacts.

        reciprocity:
            The probability `r` that a contact is reciprocated.

        dt:
            The length of an Euler step.

        substeps:
            The number of Euler steps per step of the system.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import RadicalizationInteraction
        >>> RadicalizationInteraction('influence', target='agents').contacts
        10
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        controversy: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        homophily: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        contacts: Annotated[int, ParameterSpec(domain=NN)] = 10,
        reciprocity: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
        dt: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        substeps: Annotated[int, ParameterSpec(domain=NN)] = 10,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.strength = strength
        self.controversy = controversy
        self.homophily = homophily
        self.contacts = contacts
        self.reciprocity = reciprocity
        self.dt = dt
        self.substeps = substeps

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the opinion rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.influence)]

    def _get_inputs(self, opinions: Array, activities: Array, rng: np.random.Generator) -> Array:
        """Return the summed influence `sum_j A_ij tanh(alpha x_j)` on every agent.

        The influence comes from one temporal network.
        """
        size = opinions.size
        influence = np.tanh(self.controversy * opinions)
        inputs = np.zeros(size)
        contacts = min(int(self.contacts), size - 1)
        for active in np.flatnonzero(rng.random(size) < activities).tolist():
            distances = np.abs(opinions - opinions[active])
            distances[active] = np.inf
            weights = np.maximum(distances, 1e-12) ** -self.homophily
            weights[active] = 0.0
            chosen = rng.choice(size, contacts, replace=False, p=weights / weights.sum())
            inputs[chosen] += influence[active]
            reciprocated = chosen[rng.random(contacts) < self.reciprocity]
            inputs[active] += influence[reciprocated].sum()
        return inputs

    def influence(self, system: SystemView) -> None:
        """Integrate the opinions over the Euler steps of one step of the system.

        Args:
            system:
                The system.
        """
        agents = system.agents(self.target, RadicalView)
        if len(agents) < PAIR:
            return
        opinions = np.array([agent.opinion for agent in agents])
        activities = np.array([agent.activity for agent in agents])
        for _ in range(int(self.substeps)):
            inputs = self._get_inputs(opinions, activities, system.rng)
            opinions = opinions + self.dt * (-opinions + self.strength * inputs)
        for agent, opinion in zip(agents, opinions.tolist(), strict=True):
            agent.opinion = opinion


class EchoChamberRadicalizationSystem(BaseSystem):
    """The radicalization model of echo chambers of Baumann and coauthors.

    It runs on an activity-driven network. Opinions follow `dx_i/dt = -x_i + K sum_j A_ij(t) tanh(alpha x_j)`, where the
    controversialness `alpha` shapes the influence and `A_ij(t)` is an activity-driven temporal network redrawn at every
    integration step `dt = 0.01`: agent `i` is active with the probability `a_i`, drawn from `F(a) ~ a^-gamma` on
    `[epsilon, 1]`, and contacts `m` others with the homophilic probability
    `|x_i - x_j|^-beta / sum_j |x_i - x_j|^-beta`, and a contact is reciprocated with the probability `r` (arXiv
    1906.12325 Eqs. 1 and 2). Opinions start uniformly spaced on `[-1, 1]`. For small `K alpha` they reach the neutral
    consensus, above `alpha_c = 1 / ((1 + r) K m <a>)` they radicalize, all to one side without homophily (`beta = 0`),
    and into two opposite echo chambers with it. The defaults are the polarized case of Fig. 1(c), `K = 3`, `alpha = 3`,
    `beta = 3`, `r = 0.5`, `m = 10`, and `gamma = 2.1`, with fewer agents than the 1000 of the paper so a step is fast.
    Every step of the system is ten integration steps.

    The system has the group `agents`.

    Args:
        n_agents:
            The number of agents.

        strength:
            The social interaction strength `K`.

        controversy:
            The controversialness `alpha`.

        homophily:
            The exponent `beta` of the homophily of the contacts.

        contacts:
            The number `m` of distinct agents an active agent contacts.

        reciprocity:
            The probability `r` that a contact is reciprocated.

        exponent:
            The exponent `gamma` of the power law of the activities.

        least_activity:
            The least activity `epsilon`.

        random_state:
            The seed or generator of the activities.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import EchoChamberRadicalizationSystem
        >>> EchoChamberRadicalizationSystem(50).prepare().system_.elements[0].prepare().size_
        50
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'temporal network', 'polarization', 'echo chambers')
    references = ('Baumann, Lorenz-Spreen, Sokolov, and Starnini 2020',)
    until = 100

    def __init__(
        self,
        n_agents: int = 200,
        *,
        strength: float = 3.0,
        controversy: float = 3.0,
        homophily: float = 3.0,
        contacts: int = 10,
        reciprocity: float = 0.5,
        exponent: float = 2.1,
        least_activity: float = 0.01,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.strength = strength
        self.controversy = controversy
        self.homophily = homophily
        self.contacts = contacts
        self.reciprocity = reciprocity
        self.exponent = exponent
        self.least_activity = least_activity
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        low, power = self.least_activity ** (1 - self.exponent), 1 - self.exponent
        activities = (low + rng.random(self.n_agents) * (1 - low)) ** (1 / power)
        agents = GroupElement(
            'agents',
            element=RadicalElement,
            size=self.n_agents,
            values={'opinion': np.linspace(-1.0, 1.0, self.n_agents), 'activity': activities},
        )
        influence = RadicalizationInteraction(
            'influence',
            target='agents',
            strength=self.strength,
            controversy=self.controversy,
            homophily=self.homophily,
            contacts=self.contacts,
            reciprocity=self.reciprocity,
        )
        return System(self.label, elements=[agents], interactions=[influence])
