"""Coevolving opinion networks: opinions spread along edges while discordant edges rewire to like-minded nodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Law, Symbols, Update, neighbors, rewire, uniform, where
from ....core.runtime import register_model
from ....core.space import DiscreteTime, NetworkSpace
from ....core.system import System

SMALLEST_DEGREE = 1e-9
OPINION_VISUAL = VisualSpec('sites', {'states': ['opinion'], 'low': 0.0, 'high': 1.0})


def _same(node: Symbols) -> object:
    """Return the weight of a candidate node of opinion 1 as a new end, 1 for opinion 1 and 0 otherwise."""
    return node.opinion


def _opposite(node: Symbols) -> object:
    """Return the weight of a candidate node of opinion 0 as a new end, 1 for opinion 0 and 0 otherwise."""
    return 1 - node.opinion


class Tie(Element):
    """Edge of a coevolving network, which holds no state of its own.

    Args:
        label:
            The label of the edge group.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Tie
        >>> Tie('ties').label
        'ties'
    """


class CoevolvingVoter(Element):
    """Voter of a coevolving network, with a binary opinion, its degree, and its number of discordant neighbours.

    Every step a voter is active with probability `activity` and picks a random neighbour. When the neighbour
    disagrees, the voter copies its opinion with probability `1 - rewiring`, and otherwise the interaction
    `Separation` moves the edge to a random voter who agrees with it.

    Args:
        label:
            The label of the voter or the group.

        opinion:
            The opinion, 0 or 1.

        degree:
            The number of neighbours, read at the start of every step.

        activity:
            The probability that a voter acts in a step.

        rewiring:
            The probability that an active voter with a discordant neighbour rewires instead of copying.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import CoevolvingVoter
        >>> CoevolvingVoter('voters', rewiring=0.3).rewiring
        0.3
    """

    visual: ClassVar[VisualSpec] = OPINION_VISUAL
    opinion: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    degree: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    activity: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    rewiring: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the degree and the copying of a discordant neighbour's opinion.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        degree = neighbors(1)
        discordant = neighbors(abs(s.other.opinion - s.opinion))
        copying = s.activity * (1 - s.rewiring) * discordant / (degree + SMALLEST_DEGREE)
        return [
            Update(s.degree, degree),
            Update(s.opinion, where(uniform('u') < copying, 1 - s.opinion, s.opinion)),
        ]


class Separation(Interaction):
    """Rewiring of discordant edges: the end that acts keeps the edge and moves its other end to a like-minded node.

    An end acts on an edge with probability `activity / degree` per step, and rewires a discordant edge with
    probability `rewiring`, keeping itself and drawing a new end among the voters who share its opinion.

    Args:
        label:
            The label of the interaction.

        voters:
            The path of the voters on the nodes.

        ties:
            The path of the variable group on the edges.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Separation
        >>> Separation('separation', voters='voters', ties='ties').ties
        'ties'
    """

    over: ClassVar[str] = 'neighbours'
    voters: str = role()
    ties: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rewiring of a discordant edge by either end.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        source, target = s.source, s.target
        by_source = source.activity * source.rewiring / (source.degree + SMALLEST_DEGREE)
        by_target = target.activity * target.rewiring / (target.degree + SMALLEST_DEGREE)
        return [
            Update(rewire(keep='source', toward=_same), by_source * source.opinion * (1 - target.opinion)),
            Update(rewire(keep='source', toward=_opposite), by_source * (1 - source.opinion) * target.opinion),
            Update(rewire(keep='target', toward=_same), by_target * target.opinion * (1 - source.opinion)),
            Update(rewire(keep='target', toward=_opposite), by_target * (1 - target.opinion) * source.opinion),
        ]


class ClubMember(Element):
    """Agent that is in a club (spin 1) or out of it (spin 0), aligned by a heat bath with the agents it knows.

    Every step an agent reconsiders with probability `deciding` and then joins with probability
    `1 / (1 + exp(-2 β h))`, where the field `h` counts the acquaintances in the club minus those out of it, plus the
    appeal of the club. The interaction `Separation` dissolves acquaintances between a member and an outsider and
    reforms them towards an agent of the same side.

    Args:
        label:
            The label of the agent or group.

        opinion:
            1 for a member of the club and 0 for an outsider.

        degree:
            The number of acquaintances, read at the start of every step.

        beta:
            The inverse temperature of the heat bath.

        appeal:
            The field of the club on every agent, positive for an attractive club.

        deciding:
            The probability that an agent reconsiders its choice in a step.

        activity:
            The probability that an acquaintance is reconsidered in a step.

        rewiring:
            The probability that a reconsidered acquaintance across the club boundary dissolves and reforms.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import ClubMember
        >>> ClubMember('agents', beta=2.0).beta
        2.0
    """

    visual: ClassVar[VisualSpec] = OPINION_VISUAL
    opinion: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    degree: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    appeal: float = parameter(default=0.0, domain=Domain.REAL)
    deciding: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    activity: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    rewiring: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the degree and the heat-bath choice of joining the club.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        field = neighbors(2 * s.other.opinion - 1) + s.appeal
        joining = 1 / (1 + exp(-2 * s.beta * field))
        choice = where(uniform('u') < joining, 1, 0)
        return [Update(s.degree, neighbors(1)), Update(s.opinion, where(uniform('d') < s.deciding, choice, s.opinion))]


def _coevolving(element: type[Element], space: NetworkSpace, opinions: np.ndarray, **values: object) -> System:
    """Return a coevolving network of agents with opinions and its variable ties, rewired by separation."""
    agents = Group(
        'voters',
        element=element,
        on='sites',
        variable=True,
        capacity=space.size,
        values={'opinion': opinions, **values},
        visual=OPINION_VISUAL,
    )
    ties = Group('ties', element=Tie, on='edges', variable=True)
    separation = Separation('separation', voters='voters', ties='ties')
    return System(components=[agents, ties], interactions=[separation], space=space, time=DiscreteTime())


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
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.models.social.opinion_dynamics import coevolving_voter
        >>> trajectory = Simulation(coevolving_voter(30), random_state=0).run(until=5).trajectory_
        >>> int(trajectory.network(-1).sources.size) == int(trajectory.network(0).sources.size)
        True
    """
    rng = np.random.default_rng(random_state)
    space = NetworkSpace.erdos_renyi(n_voters, mean_degree / (n_voters - 1), seed=rng)
    opinions = (rng.random(n_voters) < 0.5).astype(float)  # noqa: PLR2004
    return _coevolving(CoevolvingVoter, space, opinions, rewiring=rewiring, activity=activity)


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
        >>> join_a_club(30).components[0].element.__name__
        'ClubMember'
    """
    rng = np.random.default_rng(random_state)
    space = NetworkSpace.erdos_renyi(n_agents, mean_degree / (n_agents - 1), seed=rng)
    opinions = (rng.random(n_agents) < 0.5).astype(float)  # noqa: PLR2004
    return _coevolving(ClubMember, space, opinions, beta=beta, appeal=appeal, rewiring=rewiring)
