"""Persuasion models: agreeing pairs, majorities in small groups, and social impact."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.space import LatticeSpace, NetworkSpace
from ....core.system import System

type Array = npt.NDArray[np.float64]


def _shares(size: int, share: float, random_state: int | np.random.Generator | None) -> Array:
    """Return opinions that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(size) < share).astype(float)


class Persuader(Element):
    """Agent of the Sznajd model, who convinces the neighbours of an agreeing pair.

    A step of an agent picks a random neighbour, and when both share an opinion every neighbour of either adopts it.
    On a ring this is the original one-dimensional rule, where the pair convinces the two agents beside it.

    Args:
        label:
            The label of the agent or group.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Persuader
        >>> Persuader.group('people', 10, opinion=1.0)
        Group('people', element=Persuader, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)})
    opinion: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the persuasion rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.persuade)]

    def persuade(self, agent: AgentView, model: ModelView) -> None:
        """Convince the neighbours of the agent and a random neighbour when the two agree.

        Args:
            agent:
                The persuader.

            model:
                The model.
        """
        neighbors = agent.neighbors()
        if not neighbors:
            return
        partner = neighbors[int(model.rng.integers(len(neighbors)))]
        if partner.opinion != agent.opinion:
            return
        for other in [*neighbors, *partner.neighbors()]:
            other.opinion = agent.opinion


class Citizen(Element):
    """Agent of the Galam majority model, who holds a binary opinion.

    Args:
        label:
            The label of the agent or group.

        opinion:
            The initial opinion, 0 or 1.

        group_size:
            The number of agents in each discussion group.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Citizen
        >>> Citizen.group('citizens', 10, group_size=3)
        Group('citizens', element=Citizen, size=10)
    """

    opinion: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    group_size: float = parameter(default=3, domain=Domain.POSITIVE)


class Discussions(Interaction):
    """Discussions of the Galam model, which split the agents into random groups that adopt their majority opinion.

    A tie in a group of even size goes to opinion 0, the status quo.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of citizens.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Discussions
        >>> Discussions('talks', target='citizens').target
        'citizens'
    """

    target: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the discussion rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.discuss)]

    def discuss(self, model: ModelView) -> None:
        """Split the citizens into random groups and give each group its majority opinion.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        if not agents:
            return
        size = max(int(agents[0].group_size), 1)
        order = model.rng.permutation(len(agents))
        for start in range(0, len(agents), size):
            members = [agents[index] for index in order[start : start + size]]
            majority = float(2 * sum(member.opinion for member in members) > len(members))
            for member in members:
                member.opinion = majority


class Influencee(Element):
    """Agent of the social impact model, with an opinion of -1 or 1, a persuasiveness, and a supportiveness.

    Args:
        label:
            The label of the agent or group.

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
        >>> from skcomplex.models.social.opinion_dynamics import Influencee
        >>> Influencee.group('people', 4, opinion=-1.0)
        Group('people', element=Influencee, size=4)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (-1.0, 1.0)})
    opinion: float = state(default=1.0, domain=Domain.INTEGER)
    persuasiveness: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    supportiveness: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    exponent: float = parameter(default=2.0, domain=Domain.POSITIVE)
    noise: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)


class SocialImpact(Interaction):
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
        >>> from skcomplex.models.social.opinion_dynamics import SocialImpact
        >>> SocialImpact('impact', target='people').target
        'people'
    """

    target: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the impact rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.update)]

    def update(self, model: ModelView) -> None:
        """Flip every opinion under a positive net pressure to change.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        if len(agents) < 2:  # noqa: PLR2004
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
            The model, with the group `people`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import sznajd
        >>> sznajd(nx.cycle_graph(8), random_state=0).elements[0].size
        8
    """
    graph = nx.cycle_graph(50) if graph is None else graph
    size = nx.number_of_nodes(graph)
    people = Persuader.group('people', size, opinion=_shares(size, share, random_state))
    return System(components=[people], space=NetworkSpace(graph))


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
        >>> galam_majority(30, random_state=0).elements[0].size
        30
    """
    citizens = Citizen.group(
        'citizens',
        n_agents,
        opinion=_shares(n_agents, share, random_state),
        group_size=group_size,
    )
    return System(components=[citizens], interactions=[Discussions('talks', target='citizens')])


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
        >>> social_impact(4, 5, random_state=0).elements[0].size
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    people = Influencee.group(
        'people',
        size,
        location=[(column, row) for column in range(width) for row in range(height)],
        opinion=2 * _shares(size, share, rng) - 1,
        persuasiveness=rng.random(size),
        supportiveness=rng.random(size),
        exponent=exponent,
        noise=noise,
    )
    return System(
        components=[people],
        interactions=[SocialImpact('impact', target='people')],
        space=LatticeSpace((width, height), boundary='fixed', capacity=1),
    )
