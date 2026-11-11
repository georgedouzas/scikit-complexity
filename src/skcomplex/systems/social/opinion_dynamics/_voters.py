"""Opinion dynamics systems: voting and majority rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LocalRuleInteraction, MapInteraction
from ....core.components.spaces import BallNeighborhood, LatticeSpace, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, TotalisticSpec, VisualSpec
from ....core.types import Array, IntArray

MAJORITY = 5


class MajorityVoterElement(Element):
    """Cell that takes the majority opinion of its Moore neighbourhood.

    The neighbourhood includes the cell itself. The majority rule is a `LocalRuleInteraction` of the lattice, which
    `MajorityRuleSystem` adds.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import MajorityVoterElement
        >>> MajorityVoterElement('cell', opinion=1).opinion
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        opinion: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion


class VariantVoterView(AgentView):
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

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

        contrarian:
            Whether the agent is a contrarian, 1, or not, 0.

        threshold:
            The share of disagreeing neighbours above which the agent changes its opinion, or 0 to copy a random
            neighbour.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import VariantVoterElement
        >>> group = GroupElement('voters', element=VariantVoterElement, size=10, values={'threshold': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('voters', 'VariantVoterElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
        contrarian: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.contrarian = contrarian
        self.threshold = threshold

    @classmethod
    def _get_view(cls) -> type[VariantVoterView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return VariantVoterView


class VoterView(AgentView):
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

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import VoterElement
        >>> group = GroupElement('voters', element=VoterElement, size=20, values={'opinion': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('voters', 'VoterElement', 20)
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
    def _get_view(cls) -> type[VoterView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return VoterView


class VariantVotingInteraction(MapInteraction):
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
        >>> VariantVotingInteraction('voting', target='voters').schedule
        'random-sequential'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='opinion',
            function=self._get_opinion,
            schedule='random-sequential',
            form='agent',
        )

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
        neighbors = [item for item in agent.get_neighbors() if isinstance(item, VariantVoterView)]
        if not neighbors:
            return float(agent.opinion)
        choice = self._get_choice(agent, neighbors, system)
        return 1 - choice if agent.contrarian else choice


class VoterImitationInteraction(MapInteraction):
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
        >>> VoterImitationInteraction('imitation', target='voters').schedule
        'random-sequential'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='opinion',
            function=self._get_opinion,
            schedule='random-sequential',
            form='agent',
        )

    def _get_opinion(self, agent: VoterView, system: SystemViewProtocol) -> float:
        """Return the opinion of a random neighbouring voter.

        Without one, it is the voter's own.
        """
        neighbors = [item for item in agent.get_neighbors() if isinstance(item, VoterView)]
        if not neighbors:
            return float(agent.opinion)
        return float(neighbors[int(system.rng.integers(len(neighbors)))].opinion)


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

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import MajorityRuleSystem
        >>> MajorityRuleSystem((10, 10)).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.density = density
        self.random_state = random_state

    @staticmethod
    def _is_majority(total: IntArray) -> npt.NDArray[np.bool_]:
        """Return whether the opinions 1 of a Moore neighbourhood are a majority.

        The neighbourhood has nine cells.
        """
        return total >= MAJORITY

    @staticmethod
    def _get_random_cells(
        shape: Sequence[int],
        density: float,
        random_state: int | np.random.Generator | None,
    ) -> Array:
        """Return cells that are 1 with a probability and 0 otherwise."""
        return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        values = self._get_random_cells(self.shape, self.density, self.random_state)
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'voters',
                    element=MajorityVoterElement,
                    space=LatticeSpace(tuple(self.shape)),
                    values={'opinion': values},
                    on='sites',
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'majority',
                    target='voters',
                    state='opinion',
                    rule=TotalisticSpec(self._is_majority),
                    neighborhood=BallNeighborhood(center='first'),
                ),
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

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.opinion_dynamics import VoterSystem
        >>> VoterSystem(nx.cycle_graph(8), random_state=0).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.share = share
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
        size = nx.number_of_nodes(graph)
        opinions = (rng.random(size) < self.share).astype(float)
        return CompositeSystem(
            self.label,
            elements=[GroupElement('voters', element=VoterElement, size=size, values={'opinion': opinions})],
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

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.opinion_dynamics import VoterVariantsSystem
        >>> VoterVariantsSystem(nx.cycle_graph(8), random_state=0).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.threshold = threshold
        self.contrarians = contrarians
        self.share = share
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
        size = nx.number_of_nodes(graph)
        voters = GroupElement(
            'voters',
            element=VariantVoterElement,
            size=size,
            values={
                'opinion': (rng.random(size) < self.share).astype(float),
                'contrarian': (rng.random(size) < self.contrarians).astype(float),
                'threshold': self.threshold,
            },
        )
        return CompositeSystem(
            self.label,
            elements=[voters],
            interactions=[VariantVotingInteraction('voting', target='voters')],
            space=NetworkSpace(graph),
        )
