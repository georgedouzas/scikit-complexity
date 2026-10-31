"""Persuasion models: agreeing pairs, majorities in small groups, and social impact."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import GroupingInteraction, Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import LatticeSpace, NetworkSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._citizen_view import CitizenView
from ._influencee_view import InfluenceeView
from ._persuader_view import PersuaderView

PAIR = 2


def _shares(size: int, share: float, random_state: int | np.random.Generator | None) -> Array:
    """Return opinions that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(size) < share).astype(float)


class PersuaderElement(Element):
    """Agent of the Sznajd model, who convinces the neighbours of an agreeing pair.

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
        >>> from skcomplex.models.social.opinion_dynamics import PersuaderElement
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
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return PersuaderView


class PersuasionInteraction(GroupingInteraction):
    """Persuasion of the Sznajd model, where an agreeing pair of neighbours convinces the neighbours of both.

    Every step every agent, in random order, picks a random neighbour, and when both share an opinion every neighbour
    of either adopts it. On a ring this is the original one-dimensional rule, where the pair convinces the two agents
    beside it.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of persuaders.

        phase:
            The phase of the step the persuasion runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import PersuasionInteraction
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

    def act_on(self, members: list[AgentViewProtocol], model: ModelViewProtocol) -> None:
        """Convince the neighbours of the first member and a random neighbour of it when the two agree.

        Args:
            members:
                The persuader, followed by its neighbours.

            model:
                The model view of the rule.
        """
        agent, *neighbors = [member for member in members if isinstance(member, PersuaderView)]
        if not neighbors:
            return
        partner = neighbors[int(model.rng.integers(len(neighbors)))]
        if partner.opinion != agent.opinion:
            return
        for other in [*neighbors, *partner.neighbors()]:
            if isinstance(other, PersuaderView):
                other.opinion = agent.opinion


class CitizenElement(Element):
    """Agent of the Galam majority model, who holds a binary opinion.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import CitizenElement
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
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return CitizenView


class DiscussionsInteraction(GroupingInteraction):
    """Discussions of the Galam model, which split the agents into random groups that adopt their majority opinion.

    A tie in a group of even size goes to opinion 0, the status quo.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of citizens.

        size:
            The number of agents in each discussion group, the last group holding the remainder.

        phase:
            The phase of the step the discussions run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import DiscussionsInteraction
        >>> DiscussionsInteraction('talks', target='citizens', size=4).size
        4
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        size: int = 3,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, groups='partition', size=size, phase=phase)

    def act_on(self, members: list[AgentViewProtocol], model: ModelViewProtocol) -> None:
        """Give every member of a discussion group the majority opinion of the group.

        Args:
            members:
                The citizens of the group.

            model:
                The model view of the rule.
        """
        citizens = [member for member in members if isinstance(member, CitizenView)]
        majority = float(2 * sum(citizen.opinion for citizen in citizens) > len(citizens))
        for citizen in citizens:
            citizen.opinion = majority


class InfluenceeElement(Element):
    """Agent of the social impact model, with an opinion of -1 or 1, a persuasiveness, and a supportiveness.

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
        >>> from skcomplex.models.social.opinion_dynamics import InfluenceeElement
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
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return InfluenceeView


class SocialImpactInteraction(Interaction):
    """Synchronous update of the social impact model, where every agent takes the opinion of the stronger impact.

    The impact on agent `i` is `I_i = sum_j t_j (1 - o_i o_j) / d_ij^a - sum_j s_j (1 + o_i o_j) / d_ij^a`, with the
    persuasiveness `t`, the supportiveness `s`, and the distance `d` on the grid, where an agent supports itself from
    distance 1. The impact is the net pressure to change, whatever the opinion, so an agent flips its opinion when
    `I_i + h_i > 0`, with a uniform random field `h_i`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import SocialImpactInteraction
        >>> SocialImpactInteraction('impact', target='people').target
        'people'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the impact rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.update)]

    def update(self, model: ModelView) -> None:
        """Flip every opinion under a positive net pressure to change.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target, InfluenceeView)
        if len(agents) < PAIR:
            return
        locations = np.array([agent.location for agent in agents], dtype=float)
        opinions = np.array([agent.opinion for agent in agents], dtype=float)
        persuasion = np.array([agent.persuasiveness for agent in agents], dtype=float)
        support = np.array([agent.supportiveness for agent in agents], dtype=float)
        distances = np.linalg.norm(locations[:, None, :] - locations[None, :, :], axis=-1)
        np.fill_diagonal(distances, 1.0)
        weights = distances ** -agents[0].exponent
        agreement = opinions[:, None] * opinions[None, :]
        impact = (weights * persuasion * (1 - agreement)).sum(axis=1) - (weights * support * (1 + agreement)).sum(
            axis=1,
        )
        field = model.rng.uniform(-agents[0].noise, agents[0].noise, len(agents))
        following = np.where(impact + field > 0, -opinions, opinions)
        for agent, opinion in zip(agents, following, strict=True):
            agent.opinion = float(opinion)


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'network', 'consensus'),
    references=('Sznajd-Weron and Sznajd 2000', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 100},
)
def sznajd(
    graph: object = None,
    *,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Sznajd model, where agreeing pairs of agents convince their neighbours.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 agents.

        share:
            The share of agents with opinion 1.

        random_state:
            The seed or generator of the opinions.

    Returns:
        system:
            The model, with the group `people` and the persuasion `persuasion`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import sznajd
        >>> sznajd(nx.cycle_graph(8), random_state=0).elements[0].check().prepare().size_
        8
    """
    graph = nx.cycle_graph(50) if graph is None else graph
    size = nx.number_of_nodes(graph)
    people = GroupElement(
        'people',
        element=PersuaderElement,
        size=size,
        values={'opinion': _shares(size, share, random_state)},
    )
    return System(
        elements=[people],
        interactions=[PersuasionInteraction('persuasion', target='people')],
        space=NetworkSpace(graph),
    )


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'majority', 'consensus'),
    references=('Galam 2002', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 20},
)
def galam_majority(
    n_agents: int = 300,
    *,
    group_size: int = 3,
    share: float = 0.4,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Galam majority model, where random groups adopt their majority until one opinion wins.

    Args:
        n_agents:
            The number of agents.

        group_size:
            The number of agents in each discussion group.

        share:
            The share of agents with opinion 1.

        random_state:
            The seed or generator of the opinions.

    Returns:
        system:
            The model, with the group `citizens`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import galam_majority
        >>> galam_majority(30, random_state=0).elements[0].check().prepare().size_
        30
    """
    citizens = GroupElement(
        'citizens',
        element=CitizenElement,
        size=n_agents,
        values={'opinion': _shares(n_agents, share, random_state)},
    )
    return System(
        elements=[citizens],
        interactions=[DiscussionsInteraction('talks', target='citizens', size=group_size)],
    )


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'grid', 'clusters'),
    references=('Nowak, Szamrej, and Latané 1990', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 20},
)
def social_impact(
    width: int = 15,
    height: int = 15,
    *,
    share: float = 0.5,
    exponent: float = 2.0,
    noise: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the social impact model on a full grid, with random persuasiveness and supportiveness.

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

    Returns:
        system:
            The model, with the group `people`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import social_impact
        >>> social_impact(4, 5, random_state=0).elements[0].check().prepare().size_
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    people = GroupElement(
        'people',
        element=InfluenceeElement,
        size=size,
        values={
            'opinion': 2 * _shares(size, share, rng) - 1,
            'persuasiveness': rng.random(size),
            'supportiveness': rng.random(size),
            'exponent': exponent,
            'noise': noise,
        },
        location=[(column, row) for column in range(width) for row in range(height)],
    )
    return System(
        elements=[people],
        interactions=[SocialImpactInteraction('impact', target='people')],
        space=LatticeSpace((width, height), boundary='fixed', capacity=1),
    )
