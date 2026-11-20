"""Opinion dynamics systems: voting and majority rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import TotalisticSpec, UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace, NetworkSpace, RadiusNeighborhood
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Array, IntArray, Role

MAJORITY = 5
MOORE = RadiusNeighborhood(1, metric='chebyshev', center=True)


class MajorityVoterElement(Element):
    """Cell that takes the majority opinion of its Moore neighbourhood.

    The neighbourhood includes the cell itself. The majority rule is a `MajorityRuleInteraction` of the lattice, which
    `MajorityRuleSystem` adds.

    Args:
        label:
            The label of the cell or the lattice group.

        opinion:
            The initial opinion, 0 or 1.

        space:
            The space with one member per site, or `None` for one member.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import MajorityVoterElement
        >>> MajorityVoterElement('cell', opinion=1).opinion
        1
    """

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.opinion = opinion


class VariantVoterView(AgentViewProtocol, Protocol):
    """View of an agent of the voter system variants.

    Its states and parameters are attributes.
    """

    opinion: float
    contrarian: float
    threshold: float


class VariantVoterElement(Element):
    """Agent of the voter system variants.

    It is an imitator, a threshold voter, or a contrarian. With a threshold of 0 the voting copies a random neighbour
    into the agent, as in the voter system. With a positive threshold it gives the agent the opinion opposite to its own
    when the share of neighbours that hold it exceeds the threshold, so a threshold of 0.5 gives the majority rule. A
    contrarian adopts the opposite of what it would otherwise adopt.

    Args:
        label:
            The label of the agent or group.

        opinion:
            The initial opinion, 0 or 1.

        contrarian:
            Whether the agent is a contrarian, 1, or not, 0.

        threshold:
            The share of disagreeing neighbours above which the agent changes its opinion, or 0 to copy a random
            neighbour.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import VariantVoterElement
        >>> group = VariantVoterElement('voters', threshold=0.5, size=10)
        >>> group.label, type(group).__name__, group.size
        ('voters', 'VariantVoterElement', 10)
    """

    visual = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, NN] = 0.0,
        contrarian: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        threshold: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.opinion = opinion
        self.contrarian = contrarian
        self.threshold = threshold


class VoterView(AgentViewProtocol, Protocol):
    """View of an agent of the voter system.

    Its states and parameters are attributes.
    """

    opinion: float


class VoterElement(Element):
    """Agent of the voter system on a network.

    The imitation sets its opinion to that of a random neighbour.

    Args:
        label:
            The label of the agent or group.

        opinion:
            The initial opinion, 0 or 1.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import VoterElement
        >>> group = VoterElement('voters', opinion=1.0, size=20)
        >>> group.label, type(group).__name__, group.size
        ('voters', 'VoterElement', 20)
    """

    visual = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, NN] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.opinion = opinion


class VariantVotingInteraction(Interaction):
    """Voting of the voter system variants.

    Every voter in turn takes the opinion of its variant. A voter of threshold 0 copies a random neighbour. A voter of
    positive threshold adopts the opinion opposite to its own when the share of neighbours that hold it exceeds the
    threshold. A contrarian adopts the opposite of what it would otherwise adopt. The voters update one after the other
    in a new random order every step, each reading the opinions the voters before it set, and a voter without a
    neighbouring voter keeps its opinion.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of voters.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import VariantVotingInteraction
        >>> VariantVotingInteraction('voting', target='voters').target
        'voters'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the opinion of the agents, one after the other in a new random order, as the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self._update)]

    def _update(self, system: SystemView) -> None:
        """Set the opinion of the agents one after the other in a new random order."""
        agents = cast(list[VariantVoterView], system.get_agents('target'))
        for index in system.rng.permutation(len(agents)):
            agent = agents[int(index)]
            if agent.alive:
                agent.opinion = float(self._get_opinion(agent, system))

    def _get_choice(
        self,
        agent: VariantVoterView,
        neighbors: list[VariantVoterView],
        system: SystemViewProtocol,
    ) -> float:
        """Return the opinion a voter that conforms would take."""
        if agent.threshold <= 0:
            return float(neighbors[int(system.rng.integers(len(neighbors)))].opinion)
        disagreeing = sum(neighbor.opinion != agent.opinion for neighbor in neighbors) / len(neighbors)
        return float(1 - agent.opinion) if disagreeing > agent.threshold else float(agent.opinion)

    def _get_opinion(self, agent: VariantVoterView, system: SystemViewProtocol) -> float:
        """Return the opinion of the voter's variant, inverted for a contrarian."""
        neighbors = cast(list[VariantVoterView], [item for item in agent.get_neighbors() if item.group == agent.group])
        if not neighbors:
            return float(agent.opinion)
        choice = self._get_choice(agent, neighbors, system)
        return 1 - choice if agent.contrarian else choice


class VoterImitationInteraction(Interaction):
    """Imitation of the voter system.

    Every voter in turn adopts the opinion of a random neighbour. The voters update one after the other in a new random
    order every step, each reading the opinions the voters before it set. A voter without a neighbouring voter keeps its
    opinion.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of voters.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import VoterImitationInteraction
        >>> VoterImitationInteraction('imitation', target='voters').target
        'voters'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the opinion of the agents, one after the other in a new random order, as the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self._update)]

    def _update(self, system: SystemView) -> None:
        """Set the opinion of the agents one after the other in a new random order."""
        agents = cast(list[VoterView], system.get_agents('target'))
        for index in system.rng.permutation(len(agents)):
            agent = agents[int(index)]
            if agent.alive:
                agent.opinion = float(self._get_opinion(agent, system))

    def _get_opinion(self, agent: VoterView, system: SystemViewProtocol) -> float:
        """Return the opinion of a random neighbouring voter.

        Without one, it is the voter's own.
        """
        neighbors = cast(list[VoterView], [item for item in agent.get_neighbors() if item.group == agent.group])
        if not neighbors:
            return float(agent.opinion)
        return float(neighbors[int(system.rng.integers(len(neighbors)))].opinion)


class MajorityRuleInteraction(Interaction):
    """Majority rule of voters on a lattice: every voter takes the opinion of most of its Moore neighbourhood.

    The neighbourhood holds the voter and its eight neighbours, so it has no ties, and a voter takes opinion 1 when at
    least five of the nine hold it.

    Args:
        label:
            The label of the interaction.

        target:
            The voters whose state `opinion` the rule updates.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import MajorityRuleInteraction
        >>> MajorityRuleInteraction('majority', target='voters').target
        'voters'
    """

    def __init__(self, label: str, *, target: Role = 'voters') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _is_majority(total: IntArray) -> npt.NDArray[np.bool_]:
        """Return whether the opinions 1 of a Moore neighbourhood are a majority.

        The neighbourhood has nine cells.
        """
        return total >= MAJORITY

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the majority rule of the opinions as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            opinion = context.target.opinion
            rule = TotalisticSpec(self._is_majority)
            self.laws_: list[BaseComponent] = [UpdateLaw(opinion, rule, neighborhood=MOORE, label=self.label)]


class MajorityRuleSystem(BaseSystem):
    """Voters that take the majority opinion of their neighbourhood.

    The opinions coarsen into domains. Every step every voter takes the opinion of the majority of its eight Moore
    neighbours and keeps its own when they tie, the majority rule of Argun et al. (2021, section 4.3). The system has
    the opinions `voters.opinion`.

    Args:
        shape:
            The number of rows and columns.

        density:
            The probability that a voter starts with opinion 1.

        random_state:
            The seed or generator of the initial opinions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import MajorityRuleSystem
        >>> MajorityRuleSystem((10, 10)).prepare().system_.elements[0].opinion.size
        100
    """

    field = 'social/opinion_dynamics'
    topics = ('lattice', 'opinion', 'voting')
    references = ('Argun et al. 2021', 'Gärtner and Zehmakan 2017')
    until = 50
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.density = density
        self.random_state = random_state

    @staticmethod
    def _get_random_cells(
        shape: Sequence[int],
        density: float,
        random_state: int | np.random.Generator | None,
    ) -> Array:
        """Return cells that are 1 with a probability and 0 otherwise."""
        return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        values = self._get_random_cells(self.shape, self.density, self.random_state)
        self.system_ = System(
            self.label,
            elements=[
                MajorityVoterElement('voters', opinion=values, space=LatticeSpace(tuple(self.shape))),
            ],
            interactions=[
                MajorityRuleInteraction('majority', target='voters'),
            ],
        )


class VoterSystem(BaseSystem):
    """A voter system with one voter per node of a network.

    The system has the group `voters`.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 nodes each linked to its four nearest neighbours.

        share:
            The share of voters with opinion 1.

        random_state:
            The seed or generator of the opinions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.opinion_dynamics import VoterSystem
        >>> VoterSystem(nx.cycle_graph(8), random_state=0).prepare().system_.elements[0].size
        8
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'network')
    references = ('Holley and Liggett 1975',)
    until = 100

    def __init__(
        self,
        graph: object = None,
        *,
        share: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.share = share
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
        size = nx.number_of_nodes(graph)
        opinions = (rng.random(size) < self.share).astype(float)
        self.system_ = System(
            self.label,
            elements=[VoterElement('voters', opinion=opinions, size=size)],
            interactions=[VoterImitationInteraction('imitation', target='voters')],
            space=NetworkSpace(graph),
        )


class VoterVariantsSystem(BaseSystem):
    """Voter system variants on a network.

    They have threshold voters and a share of contrarians. The system has the group `voters`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.opinion_dynamics import VoterVariantsSystem
        >>> VoterVariantsSystem(nx.cycle_graph(8), random_state=0).prepare().system_.elements[0].size
        8
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'network', 'voting', 'contrarians')
    references = ('Thurner, Hanel, and Klimek 2018', 'Galam 2004')
    until = 100

    def __init__(
        self,
        graph: object = None,
        *,
        threshold: float = 0.5,
        contrarians: float = 0.1,
        share: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.threshold = threshold
        self.contrarians = contrarians
        self.share = share
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
        size = nx.number_of_nodes(graph)
        voters = VariantVoterElement(
            'voters',
            opinion=(rng.random(size) < self.share).astype(float),
            contrarian=(rng.random(size) < self.contrarians).astype(float),
            threshold=self.threshold,
            size=size,
        )
        self.system_ = System(
            self.label,
            elements=[voters],
            interactions=[VariantVotingInteraction('voting', target='voters')],
            space=NetworkSpace(graph),
        )
