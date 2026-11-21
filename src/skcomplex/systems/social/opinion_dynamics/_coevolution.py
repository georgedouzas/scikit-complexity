"""Coevolving opinion networks: opinions spread along edges as the edges rewire.

The discordant edges rewire to like-minded nodes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import neighbors, uniform, where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role

OPINION_VISUAL = VisualSpec('sites', {'states': ['opinion'], 'low': 0.0, 'high': 1.0})
SMALLEST_DEGREE = 1e-9
FIRST_OPINION_SHARE = 0.5


class CoevolvingVoterElement(Element):
    """Voter of a coevolving network, with a binary opinion and its degree.

    It also holds its number of discordant neighbours. Every step a voter is active with probability `activity` and
    picks a random neighbour. When the neighbour disagrees, the voter copies its opinion with probability
    `1 - rewiring`, and otherwise the interaction `SeparationInteraction` moves the edge to a random voter who agrees
    with it.

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

        space:
            The network whose nodes hold the voters, or `None` for one voter.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import CoevolvingVoterElement
        >>> CoevolvingVoterElement('voters', rewiring=0.3).rewiring
        0.3
    """

    visual = OPINION_VISUAL

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        degree: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        activity: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        rewiring: Annotated[object, RealSet.closed_open(0, oo)] = 0.3,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.opinion = opinion
        self.degree = degree
        self.activity = activity
        self.rewiring = rewiring


class ClubMemberElement(Element):
    """Agent that is in a club (spin 1) or out of it (spin 0).

    A heat bath aligns it with the agents it knows. Every step an agent reconsiders with probability `deciding` and then
    joins with probability `1 / (1 + exp(-2 β h))`, where the field `h` counts the acquaintances in the club minus those
    out of it, plus the appeal of the club. The interaction `SeparationInteraction` dissolves acquaintances between a
    member and an outsider and reforms them towards an agent of the same side.

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
        >>> from skcomplex.systems.social.opinion_dynamics import ClubMemberElement
        >>> ClubMemberElement('agents', beta=2.0).beta
        2.0
    """

    visual = OPINION_VISUAL

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        degree: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        beta: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        appeal: object = 0.0,
        deciding: Annotated[object, RealSet.closed_open(0, oo)] = 0.1,
        activity: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        rewiring: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.opinion = opinion
        self.degree = degree
        self.beta = beta
        self.appeal = appeal
        self.deciding = deciding
        self.activity = activity
        self.rewiring = rewiring


class SeparationInteraction(Interaction):
    """Rewiring of discordant edges by one end.

    It keeps the edge and moves its other end to a like-minded node, a request to rewire the edges of the voters. The
    end `keep` acts on an edge with probability `activity / degree` per step, and when it holds the opinion `opinion`
    and the other end does not, it rewires the edge with probability `rewiring`, drawing a new end among the voters of
    the opinion `opinion`.

    Args:
        label:
            The label of the interaction.

        among:
            The path of the voters on the nodes, whose edges the laws run on.

        keep:
            The end of an edge that acts and keeps the edge, `'source'` or `'target'`.

        opinion:
            The opinion of the end that acts, 0 or 1.

        along:
            The edges the laws run on, `True` for the edges of the network of the voters.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import SeparationInteraction
        >>> SeparationInteraction('separation', among='voters', keep='target', opinion=0).keep
        'target'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        among: Role,
        keep: str = 'source',
        opinion: float = 1.0,
        along: bool | str = True,
    ) -> None:
        super().__init__(label, among=among, along=along)
        self.keep = keep
        self.opinion = opinion

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the request to rewire an edge, with the probability of its acting end, as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        request = TopologyRequestSpec('rewire', toward=self._get_weight, keep=self.keep)
        self.laws_: list[BaseComponent] = [UpdateLaw(request, self._get_chance(context))]

    def _get_weight(self, node: SimpleNamespace) -> object:
        """Return the weight of a candidate node as a new end.

        It is 1 when it holds the opinion of the acting end.
        """
        return node.opinion if self.opinion == 1 else 1 - node.opinion

    def _get_chance(self, namespace: SimpleNamespace) -> object:
        """Return the probability that the acting end rewires the edge.

        The edge is rewired when the ends hold different opinions.
        """
        actor, other = (namespace.source, namespace.target)[:: 1 if self.keep == 'source' else -1]
        chance = actor.activity * actor.rewiring / (actor.degree + SMALLEST_DEGREE)
        if self.opinion == 1:
            return chance * actor.opinion * (1 - other.opinion)
        return chance * (1 - actor.opinion) * other.opinion


class CoevolvingVoterSystem(BaseSystem):
    """The coevolving voter system.

    It fragments into consensual components when rewiring is fast. Voters on an Erdős-Rényi graph copy discordant
    neighbours or rewire the discordant edges to like-minded voters. Below a critical rewiring probability, near 0.38
    for mean degree 4, the network stays connected and discordant edges survive for long times, and above it the network
    splits into components of one opinion. The system has the groups `voters` and `ties` and the interaction
    `separation`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import CoevolvingVoterSystem
        >>> trajectory = CoevolvingVoterSystem(30).prepare().simulate(until=5, random_state=0)
        >>> int(trajectory.get_network(-1).sources.size) == int(trajectory.get_network(0).sources.size)
        True
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'voter', 'adaptive network', 'rewiring', 'fragmentation')
    references = ('Holme and Newman 2006', 'Vazquez, Eguíluz, and San Miguel 2008', 'Thurner et al. 2018')
    until = 200

    @staticmethod
    def _count_neighbors(namespace: SimpleNamespace) -> object:
        """Return the number of neighbours of a voter."""
        return neighbors(1)

    @staticmethod
    def _get_coevolving(
        label: str,
        element: type[CoevolvingVoterElement | ClubMemberElement],
        space: NetworkSpace,
        opinions: np.ndarray,
        choice: Interaction,
        **values: object,
    ) -> System:
        """Return a coevolving network of agents with opinions and its variable ties.

        The ties are rewired by separation.
        """
        agents = element('voters', opinion=opinions, space=space, **values)
        ties = Element('ties', space=space, on='edges')
        interactions = [
            Interaction(
                'degree',
                roles={'target': 'voters'},
                laws=lambda s: [UpdateLaw(s.target.degree, CoevolvingVoterSystem._count_neighbors(s))],
            ),
            choice,
            *(
                SeparationInteraction(f'separation_{keep}_{int(opinion)}', among='voters', keep=keep, opinion=opinion)
                for keep in ('source', 'target')
                for opinion in (1.0, 0.0)
            ),
        ]
        return System(label, elements=[agents, ties], interactions=interactions, space=space)

    def __init__(
        self,
        n_voters: int = 200,
        *,
        mean_degree: float = 4.0,
        rewiring: float = 0.3,
        activity: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_voters = n_voters
        self.mean_degree = mean_degree
        self.rewiring = rewiring
        self.activity = activity
        self.random_state = random_state

    @staticmethod
    def _copy_discordant(namespace: SimpleNamespace) -> object:
        """Return the opinion of a voter after it copies a random discordant neighbour.

        It copies with its probability.
        """
        voter = namespace.target
        discordant = neighbors(abs(voter.other.opinion - voter.opinion))
        copying = voter.activity * (1 - voter.rewiring) * discordant / (neighbors(1) + SMALLEST_DEGREE)
        return where(uniform('u') < copying, 1 - voter.opinion, voter.opinion)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        space = NetworkSpace(nx.gnp_random_graph(self.n_voters, self.mean_degree / (self.n_voters - 1), seed=rng))
        opinions = (rng.random(self.n_voters) < FIRST_OPINION_SHARE).astype(float)
        copying = Interaction(
            'copying',
            roles={'target': 'voters'},
            laws=lambda s: [UpdateLaw(s.target.opinion, self._copy_discordant(s))],
        )
        self.system_ = CoevolvingVoterSystem._get_coevolving(
            self.label,
            CoevolvingVoterElement,
            space,
            opinions,
            copying,
            rewiring=self.rewiring,
            activity=self.activity,
        )


class JoinAClubSystem(BaseSystem):
    """Join-a-club spins.

    Agents join a club with their acquaintances while acquaintances follow the club. Agents of an Erdős-Rényi graph are
    in or out of a club, a spin, and choose by a heat bath with the inverse temperature `beta` in the field of their
    acquaintances and of the appeal of the club. Acquaintances across the club boundary dissolve and reform among agents
    on the same side. Without coupling the choices stay random and many acquaintances cross the boundary, and with
    strong coupling the members and the outsiders separate into communities with few acquaintances between them. The
    system has the groups `voters` and `ties` and the interaction `separation`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import JoinAClubSystem
        >>> type(JoinAClubSystem(30).prepare().system_.elements[0]).__name__
        'ClubMemberElement'
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'spins', 'clubs', 'adaptive network', 'heat bath')
    references = ('Thurner et al. 2018', 'Biely, Hanel, and Thurner 2009')
    until = 150

    def __init__(
        self,
        n_agents: int = 200,
        *,
        mean_degree: float = 6.0,
        beta: float = 0.5,
        appeal: float = 0.0,
        rewiring: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.mean_degree = mean_degree
        self.beta = beta
        self.appeal = appeal
        self.rewiring = rewiring
        self.random_state = random_state

    @staticmethod
    def _choose_club(namespace: SimpleNamespace) -> object:
        """Return the choice of an agent that reconsiders.

        It joins by a heat bath in the field of its acquaintances.
        """
        agent = namespace.target
        field = neighbors(2 * agent.other.opinion - 1) + agent.appeal
        joining = 1 / (1 + exp(-2 * agent.beta * field))
        choice = where(uniform('u') < joining, 1, 0)
        return where(uniform('d') < agent.deciding, choice, agent.opinion)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        space = NetworkSpace(nx.gnp_random_graph(self.n_agents, self.mean_degree / (self.n_agents - 1), seed=rng))
        opinions = (rng.random(self.n_agents) < FIRST_OPINION_SHARE).astype(float)
        choice = Interaction(
            'choice',
            roles={'target': 'voters'},
            laws=lambda s: [UpdateLaw(s.target.opinion, self._choose_club(s))],
        )
        self.system_ = CoevolvingVoterSystem._get_coevolving(
            self.label,
            ClubMemberElement,
            space,
            opinions,
            choice,
            beta=self.beta,
            appeal=self.appeal,
            rewiring=self.rewiring,
        )
