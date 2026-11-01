"""Persuasion models: agreeing pairs, majorities in small groups, and social impact."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import networkx as nx
import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import GroupingInteraction
from ....core.spaces import LatticeSpace, NetworkSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec
from ....types import Array
from ..._catalog import register_model
from ._citizen import CitizenElement
from ._discussions import DiscussionsInteraction
from ._influencee import InfluenceeElement
from ._persuader import PersuaderElement
from ._persuader_view import PersuaderView
from ._social_impact import SocialImpactInteraction


def _shares(size: int, share: float, random_state: int | np.random.Generator | None) -> Array:
    """Return opinions that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(size) < share).astype(float)


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
