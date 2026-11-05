"""Coevolving opinion networks: opinions spread along edges while discordant edges rewire to like-minded nodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import exp

from ....core.elements import Element, GroupElement
from ....core.interactions import MapInteraction
from ....core.laws.utils import neighbors, uniform, where
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ..._catalog import register_model
from ._club_member import ClubMemberElement
from ._coevolving_voter import OPINION_VISUAL, CoevolvingVoterElement
from ._separation import SMALLEST_DEGREE, SeparationInteraction
from ._tie import TieElement

FIRST_OPINION_SHARE = 0.5


def _count_neighbors(namespace: SimpleNamespace) -> object:
    """Return the number of neighbours of a voter."""
    return neighbors(1)


def _copy_discordant(namespace: SimpleNamespace) -> object:
    """Return the opinion of a voter after it copies a random discordant neighbour, with its probability."""
    voter = namespace.target
    discordant = neighbors(abs(voter.other.opinion - voter.opinion))
    copying = voter.activity * (1 - voter.rewiring) * discordant / (neighbors(1) + SMALLEST_DEGREE)
    return where(uniform('u') < copying, 1 - voter.opinion, voter.opinion)


def _choose_club(namespace: SimpleNamespace) -> object:
    """Return the choice of an agent that reconsiders, joining by a heat bath in the field of its acquaintances."""
    agent = namespace.target
    field = neighbors(2 * agent.other.opinion - 1) + agent.appeal
    joining = 1 / (1 + exp(-2 * agent.beta * field))
    choice = where(uniform('u') < joining, 1, 0)
    return where(uniform('d') < agent.deciding, choice, agent.opinion)


def _coevolving(
    element: type[Element],
    space: NetworkSpace,
    opinions: np.ndarray,
    choice: MapInteraction,
    **values: object,
) -> System:
    """Return a coevolving network of agents with opinions and its variable ties, rewired by separation."""
    agents = GroupElement(
        'voters',
        element=element,
        on='sites',
        variable=True,
        capacity=space.prepare().size_,
        values={'opinion': opinions, **values},
        visual=OPINION_VISUAL,
    )
    ties = GroupElement('ties', element=TieElement, on='edges', variable=True)
    interactions = [
        MapInteraction('degree', target='voters', state='degree', function=_count_neighbors),
        choice,
        *(
            SeparationInteraction(f'separation_{keep}_{int(opinion)}', voters='voters', keep=keep, opinion=opinion)
            for keep in ('source', 'target')
            for opinion in (1.0, 0.0)
        ),
    ]
    return System(elements=[agents, ties], interactions=interactions, space=space, time=DiscreteTime())


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'voter', 'adaptive network', 'rewiring', 'fragmentation'),
    references=('Holme and Newman 2006', 'Vazquez, Eguíluz, and San Miguel 2008', 'Thurner et al. 2018'),
    defaults={'until': 200},
)
def coevolving_voter(
    n_voters: int = 200,
    *,
    mean_degree: float = 4.0,
    rewiring: float = 0.3,
    activity: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the coevolving voter model, which fragments into consensual components when rewiring is fast.

    Voters on an Erdős-Rényi graph copy discordant neighbours or rewire the discordant edges to like-minded voters.
    Below a critical rewiring probability, near 0.38 for mean degree 4, the network stays connected and discordant
    edges survive for long times, and above it the network splits into components of one opinion.

    Args:
        n_voters:
            The number of voters.

        mean_degree:
            The mean degree of the initial random graph.

        rewiring:
            The probability that an active voter with a discordant neighbour rewires instead of copying.

        activity:
            The probability that a voter acts in a step.

        random_state:
            The seed or generator of the graph and of the uniform random opinions.

    Returns:
        system:
            The model, with the groups `voters` and `ties` and the interaction `separation`.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.social.opinion_dynamics import coevolving_voter
        >>> trajectory = Simulation(coevolving_voter(30), random_state=0).prepare().run(until=5).trajectory_
        >>> int(trajectory.network(-1).sources.size) == int(trajectory.network(0).sources.size)
        True
    """
    rng = np.random.default_rng(random_state)
    space = NetworkSpace(nx.gnp_random_graph(n_voters, mean_degree / (n_voters - 1), seed=rng))
    opinions = (rng.random(n_voters) < FIRST_OPINION_SHARE).astype(float)
    copying = MapInteraction('copying', target='voters', state='opinion', function=_copy_discordant)
    return _coevolving(CoevolvingVoterElement, space, opinions, copying, rewiring=rewiring, activity=activity)


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'spins', 'clubs', 'adaptive network', 'heat bath'),
    references=('Thurner et al. 2018', 'Biely, Hanel, and Thurner 2009'),
    defaults={'until': 150},
)
def join_a_club(
    n_agents: int = 200,
    *,
    mean_degree: float = 6.0,
    beta: float = 0.5,
    appeal: float = 0.0,
    rewiring: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return join-a-club spins, agents who join a club with their acquaintances while acquaintances follow the club.

    Agents of an Erdős-Rényi graph are in or out of a club, a spin, and choose by a heat bath with the inverse
    temperature `beta` in the field of their acquaintances and of the appeal of the club. Acquaintances across the
    club boundary dissolve and reform among agents on the same side. Without coupling the choices stay random and
    many acquaintances cross the boundary, and with strong coupling the members and the outsiders separate into
    communities with few acquaintances between them.

    Args:
        n_agents:
            The number of agents.

        mean_degree:
            The mean number of acquaintances.

        beta:
            The inverse temperature of the heat bath.

        appeal:
            The field of the club on every agent.

        rewiring:
            The probability that a reconsidered acquaintance across the club boundary reforms on one side.

        random_state:
            The seed or generator of the graph and of the uniform random choices.

    Returns:
        system:
            The model, with the groups `voters` and `ties` and the interaction `separation`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import join_a_club
        >>> join_a_club(30).elements[0].element.__name__
        'ClubMemberElement'
    """
    rng = np.random.default_rng(random_state)
    space = NetworkSpace(nx.gnp_random_graph(n_agents, mean_degree / (n_agents - 1), seed=rng))
    opinions = (rng.random(n_agents) < FIRST_OPINION_SHARE).astype(float)
    choice = MapInteraction('choice', target='voters', state='opinion', function=_choose_club)
    return _coevolving(ClubMemberElement, space, opinions, choice, beta=beta, appeal=appeal, rewiring=rewiring)
