"""Persuasion systems: agreeing pairs, majorities in small groups, and impact.

The impact is social.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import GroupingInteraction, MapInteraction
from ....core.components.laws.utils import mean, where
from ....core.components.spaces import LatticeSpace, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class PersuaderView(AgentView):
    """View of an agent of the Sznajd system.

    Its states and parameters are attributes.
    """

    opinion: float


class PersuaderElement(Element):
    """Agent of the Sznajd system, who convinces the neighbours of an agreeing pair.

    A step of an agent picks a random neighbour, and when both share an opinion every neighbour of either adopts it.
    On a ring this is the original one-dimensional rule, where the pair convinces the two agents beside it.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import PersuaderElement
        >>> group = GroupElement('people', element=PersuaderElement, size=10, values={'opinion': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('people', 'PersuaderElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion

    @classmethod
    def _get_view(cls) -> type[PersuaderView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return PersuaderView


class CitizenView(AgentView):
    """View of an agent of the Galam majority system.

    Its states and parameters are attributes.
    """

    opinion: float


class CitizenElement(Element):
    """Agent of the Galam majority system, who holds a binary opinion.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import CitizenElement
        >>> group = GroupElement('citizens', element=CitizenElement, size=10, values={'opinion': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('citizens', 'CitizenElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion

    @classmethod
    def _get_view(cls) -> type[CitizenView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return CitizenView


class InfluenceeView(AgentView):
    """View of an agent of the social impact system.

    Its states and parameters are attributes.
    """

    opinion: float
    persuasiveness: float
    supportiveness: float
    exponent: float
    noise: float


class InfluenceeElement(Element):
    """Agent of the social impact system.

    It holds an opinion of -1 or 1, a persuasiveness, and a supportiveness.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, -1 or 1.

        persuasiveness:
            The strength with which the agent persuades agents of the other opinion.

        supportiveness:
            The strength with which the agent supports agents of its opinion.

        exponent:
            The exponent of the distance by which impact decays.

        noise:
            The half-width of the uniform random field added to the impact on every agent and step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import InfluenceeElement
        >>> group = GroupElement('people', element=InfluenceeElement, size=4, values={'opinion': -1.0})
        >>> group.label, group.element.__name__, group.size
        ('people', 'InfluenceeElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (-1.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        persuasiveness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        supportiveness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        exponent: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.persuasiveness = persuasiveness
        self.supportiveness = supportiveness
        self.exponent = exponent
        self.noise = noise

    @classmethod
    def _get_view(cls) -> type[InfluenceeView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return InfluenceeView


class DiscussionsInteraction(GroupingInteraction):
    """Discussions of the Galam system.

    They split the citizens into random groups that adopt their majority opinion. It is a symbolic `GroupingInteraction`
    of random partitions, drawn again at every step. A tie in a group of even size goes to opinion 0, the status quo.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of citizens.

        size:
            The number of citizens in each discussion group, the last group holding the remainder.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import DiscussionsInteraction
        >>> DiscussionsInteraction('talks', target='citizens', size=4).write
        'opinion'
    """

    draws = True

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], size: int = 3) -> None:
        super().__init__(
            label,
            target=target,
            groups='partition',
            size=size,
            form='symbolic',
            write='opinion',
            function=self._get_majority,
        )

    def _get_majority(self, group: SimpleNamespace) -> Expression:
        """Return 1 when most members of a group hold opinion 1, and 0 otherwise.

        Most means more than half.
        """
        return where(mean(group.members.opinion, over=group.members) > 1 / 2, 1, 0)


class SocialImpactInteraction(MapInteraction):
    """Synchronous update of the social impact system.

    Every agent takes the opinion of the stronger impact. The impact on agent `i` is
    `I_i = sum_j t_j (1 - o_i o_j) / d_ij^a - sum_j s_j (1 + o_i o_j) / d_ij^a`, with the persuasiveness `t`, the
    supportiveness `s`, and the distance `d` on the grid, where an agent supports itself from distance 1. The impact is
    the net pressure to change, whatever the opinion, so an agent flips its opinion when `I_i + h_i > 0`, with a uniform
    random field `h_i`. Every agent decides from the opinions at the start of the step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import SocialImpactInteraction
        >>> SocialImpactInteraction('impact', target='people').form
        'agent'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label, target=target, state='opinion', function=self._get_opinion, form='agent')

    def _get_impact(self, agent: InfluenceeView, system: SystemViewProtocol) -> float:
        """Return the net pressure to change on an agent from every agent of the group.

        The agent itself counts from distance 1.
        """
        agents = [item for item in system.get_agents('target') if isinstance(item, InfluenceeView)]
        locations = np.array([item.location for item in agents], dtype=float)
        distances = np.linalg.norm(locations - np.asarray(agent.location, dtype=float), axis=-1)
        distances[[item.id == agent.id for item in agents]] = 1.0
        weights = distances**-agent.exponent
        agreement = agent.opinion * np.array([item.opinion for item in agents], dtype=float)
        persuasion = np.array([item.persuasiveness for item in agents], dtype=float)
        support = np.array([item.supportiveness for item in agents], dtype=float)
        return float((weights * persuasion * (1 - agreement)).sum() - (weights * support * (1 + agreement)).sum())

    def _get_opinion(self, agent: InfluenceeView, system: SystemViewProtocol) -> float:
        """Return the opposite opinion of an agent under a positive net pressure to change.

        Otherwise it is its own.
        """
        field = system.rng.uniform(-agent.noise, agent.noise)
        return -agent.opinion if self._get_impact(agent, system) + field > 0 else float(agent.opinion)


class PersuasionInteraction(GroupingInteraction):
    """Persuasion of the Sznajd system.

    An agreeing pair of neighbours convinces the neighbours of both. Every step every agent, in random order, picks a
    random neighbour, and when both share an opinion every neighbour of either adopts it. On a ring this is the original
    one-dimensional rule, where the pair convinces the two agents beside it.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of persuaders.

        phase:
            The phase of the step the persuasion runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import PersuasionInteraction
        >>> PersuasionInteraction('persuasion', target='people').groups
        'neighborhood'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, groups='neighborhood', phase=phase)

    def act_on(self, members: list[AgentViewProtocol], system: SystemViewProtocol) -> None:
        """Convince the neighbours of the first member and of a random neighbour of it.

        This happens when the two agree.

        Args:
            members:
                The persuader, followed by its neighbours.

            system:
                The system view of the rule.
        """
        agent, *neighbors = [member for member in members if isinstance(member, PersuaderView)]
        if not neighbors:
            return
        partner = neighbors[int(system.rng.integers(len(neighbors)))]
        if partner.opinion != agent.opinion:
            return
        for other in [*neighbors, *partner.get_neighbors()]:
            if isinstance(other, PersuaderView):
                other.opinion = agent.opinion


class SznajdSystem(BaseSystem):
    """The Sznajd system, where agreeing pairs of agents convince their neighbours.

    The system has the group `people` and the persuasion `persuasion`.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 agents.

        share:
            The share of agents with opinion 1.

        random_state:
            The seed or generator of the opinions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.opinion_dynamics import SznajdSystem
        >>> SznajdSystem(nx.cycle_graph(8), random_state=0).prepare().system_.elements[0].prepare().size_
        8
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'network', 'consensus')
    references = ('Sznajd-Weron and Sznajd 2000', 'Castellano, Fortunato, and Loreto 2009')
    until = 100

    @staticmethod
    def _get_shares(size: int, share: float, random_state: int | np.random.Generator | None) -> Array:
        """Return opinions that are 1 with a probability and 0 otherwise."""
        return (np.random.default_rng(random_state).random(size) < share).astype(float)

    def __init__(
        self,
        graph: object = None,
        *,
        share: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.share = share
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        graph = self.graph
        graph = nx.cycle_graph(50) if graph is None else graph
        size = nx.number_of_nodes(graph)
        people = GroupElement(
            'people',
            element=PersuaderElement,
            size=size,
            values={'opinion': SznajdSystem._get_shares(size, self.share, self.random_state)},
        )
        return CompositeSystem(
            self.label,
            elements=[people],
            interactions=[PersuasionInteraction('persuasion', target='people')],
            space=NetworkSpace(graph),
        )


class GalamMajoritySystem(BaseSystem):
    """The Galam majority system.

    Random groups adopt their majority until one opinion wins. The system has the group `citizens`.

    Args:
        n_agents:
            The number of agents.

        group_size:
            The number of agents in each discussion group.

        share:
            The share of agents with opinion 1.

        random_state:
            The seed or generator of the opinions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import GalamMajoritySystem
        >>> GalamMajoritySystem(30, random_state=0).prepare().system_.elements[0].prepare().size_
        30
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'majority', 'consensus')
    references = ('Galam 2002', 'Castellano, Fortunato, and Loreto 2009')
    until = 20

    def __init__(
        self,
        n_agents: int = 300,
        *,
        group_size: int = 3,
        share: float = 0.4,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.group_size = group_size
        self.share = share
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        citizens = GroupElement(
            'citizens',
            element=CitizenElement,
            size=self.n_agents,
            values={'opinion': SznajdSystem._get_shares(self.n_agents, self.share, self.random_state)},
        )
        return CompositeSystem(
            self.label,
            elements=[citizens],
            interactions=[DiscussionsInteraction('talks', target='citizens', size=self.group_size)],
        )


class SocialImpactSystem(BaseSystem):
    """The social impact system on a full grid.

    The persuasiveness and the supportiveness are random. The system has the group `people`.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        share:
            The share of agents with opinion 1, the others holding -1.

        exponent:
            The exponent of the distance by which impact decays.

        noise:
            The half-width of the uniform random field.

        random_state:
            The seed or generator of the opinions and strengths.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import SocialImpactSystem
        >>> SocialImpactSystem(4, 5, random_state=0).prepare().system_.elements[0].prepare().size_
        20
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'grid', 'clusters')
    references = ('Nowak, Szamrej, and Latané 1990', 'Castellano, Fortunato, and Loreto 2009')
    until = 20

    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        *,
        share: float = 0.5,
        exponent: float = 2.0,
        noise: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.share = share
        self.exponent = exponent
        self.noise = noise
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        size = self.width * self.height
        people = GroupElement(
            'people',
            element=InfluenceeElement,
            size=size,
            values={
                'opinion': 2 * SznajdSystem._get_shares(size, self.share, rng) - 1,
                'persuasiveness': rng.random(size),
                'supportiveness': rng.random(size),
                'exponent': self.exponent,
                'noise': self.noise,
            },
            location=[(column, row) for column in range(self.width) for row in range(self.height)],
        )
        return CompositeSystem(
            self.label,
            elements=[people],
            interactions=[SocialImpactInteraction('impact', target='people')],
            space=LatticeSpace((self.width, self.height), boundary='fixed', capacity=1),
        )
