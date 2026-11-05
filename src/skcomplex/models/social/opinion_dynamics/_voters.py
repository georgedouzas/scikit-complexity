"""Opinion dynamics models: voting and majority rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx
import numpy as np

from ....core.elements import GroupElement, LatticeElement
from ....core.interactions import LocalRuleInteraction
from ....core.spaces import LatticeSpace, MooreNeighborhood, NetworkSpace
from ....core.system import System
from ....specs import TotalisticSpec
from ....types import Array
from ..._catalog import register_model
from ._majority_voter import MajorityVoterElement
from ._variant_voter import VariantVoterElement
from ._variant_voting import VariantVotingInteraction
from ._voter import VoterElement
from ._voter_imitation import VoterImitationInteraction

MAJORITY = 5


def _is_majority(total: float) -> bool:
    """Return whether the opinions 1 of the nine cells of a Moore neighbourhood are a majority."""
    return total >= MAJORITY


def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)


@register_model(
    field='social/opinion_dynamics',
    tags=('lattice', 'opinion', 'voting'),
    references=('Charbonneau 2017',),
    defaults={'until': 50, 'dt': 1},
)
def majority_rule(
    shape: Sequence[int] = (100, 100),
    *,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return voters that take the majority opinion of their neighbourhood, which coarsens into domains.

    Args:
        shape:
            The number of rows and columns.

        density:
            The probability that a voter starts with opinion 1.

        random_state:
            The seed or generator of the initial opinions.

    Returns:
        system:
            The model, with the opinions `voters.opinion`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import majority_rule
        >>> majority_rule((10, 10)).elements[0].prepare().size_
        100
    """
    values = _random_cells(shape, density, random_state)
    return System(
        elements=[
            LatticeElement(
                'voters',
                element=MajorityVoterElement,
                space=LatticeSpace(tuple(shape)),
                values={'opinion': values},
            ),
        ],
        interactions=[
            LocalRuleInteraction(
                'majority',
                target='voters',
                state='opinion',
                rule=TotalisticSpec(_is_majority),
                neighborhood=MooreNeighborhood(center=True),
            ),
        ],
    )


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'network'),
    references=('Holley and Liggett 1975',),
    defaults={'until': 100},
)
def voter_model(
    graph: object = None,
    *,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a voter model with one voter per node of a network.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 nodes each linked to its four nearest neighbours.

        share:
            The share of voters with opinion 1.

        random_state:
            The seed or generator of the opinions.

    Returns:
        system:
            The model, with the group `voters`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import voter_model
        >>> voter_model(nx.cycle_graph(8), random_state=0).elements[0].prepare().size_
        8
    """
    rng = np.random.default_rng(random_state)
    graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
    size = nx.number_of_nodes(graph)
    opinions = (rng.random(size) < share).astype(float)
    return System(
        elements=[GroupElement('voters', element=VoterElement, size=size, values={'opinion': opinions})],
        interactions=[VoterImitationInteraction('imitation', target='voters')],
        space=NetworkSpace(graph),
    )


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'network', 'voting', 'contrarians'),
    references=('Thurner, Hanel, and Klimek 2018', 'Galam 2004'),
    defaults={'until': 100},
)
def voter_variants(
    graph: object = None,
    *,
    threshold: float = 0.5,
    contrarians: float = 0.1,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return voter model variants on a network, with threshold voters and a share of contrarians.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 nodes each linked to its four nearest neighbours.

        threshold:
            The share of disagreeing neighbours above which a voter changes its opinion, 0.5 for the majority rule, or
            0 to copy a random neighbour.

        contrarians:
            The share of voters that are contrarians.

        share:
            The share of voters with opinion 1.

        random_state:
            The seed or generator of the opinions and of the contrarians.

    Returns:
        system:
            The model, with the group `voters`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import voter_variants
        >>> voter_variants(nx.cycle_graph(8), random_state=0).elements[0].prepare().size_
        8
    """
    rng = np.random.default_rng(random_state)
    graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
    size = nx.number_of_nodes(graph)
    voters = GroupElement(
        'voters',
        element=VariantVoterElement,
        size=size,
        values={
            'opinion': (rng.random(size) < share).astype(float),
            'contrarian': (rng.random(size) < contrarians).astype(float),
            'threshold': threshold,
        },
    )
    return System(
        elements=[voters],
        interactions=[VariantVotingInteraction('voting', target='voters')],
        space=NetworkSpace(graph),
    )
