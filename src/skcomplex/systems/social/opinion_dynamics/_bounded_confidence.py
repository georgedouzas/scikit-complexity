"""Bounded confidence systems: compromise between agents whose opinions are close, in pairs or all at once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import CouplingInteraction
from ....core.components.laws.utils import where
from ....core.components.spaces import AdjacencyNeighborhood, DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ._random_pairs import RandomPairsInteraction


class ConfidentElement(Element):
    """Agent of the Hegselmann-Krause system, with an opinion and the confidence within which it trusts neighbours.

    `ConfidenceCouplingInteraction` moves every agent each step to the mean opinion of the neighbours, itself
    included, whose opinions differ from its own by less than `confidence`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The opinion, between 0 and 1.

        confidence:
            The largest opinion difference of a neighbour an agent still listens to.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ConfidentElement
        >>> ConfidentElement('people', confidence=0.1).confidence
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('histograms', {'color_range': (0.0, 1.0), 'histogram_first': True}),
        *,
        opinion: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        confidence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.confidence = confidence


class DiscussantView(AgentView):
    """View of an agent of the Deffuant bounded confidence system, with its states and parameters as attributes."""

    opinion: float
    confidence: float
    convergence: float


class DiscussantElement(Element):
    """Agent of the Deffuant system, with a continuous opinion and a confidence bound.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, between 0 and 1.

        confidence:
            The largest opinion difference at which two agents still compromise.

        convergence:
            The fraction of the opinion difference by which each agent of a compromising pair moves.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import DiscussantElement
        >>> group = GroupElement('people', element=DiscussantElement, size=10, values={'opinion': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('people', 'DiscussantElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        confidence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        convergence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.confidence = confidence
        self.convergence = convergence

    @classmethod
    def _get_view(cls) -> type[DiscussantView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return DiscussantView


class ConfidenceCouplingInteraction(CouplingInteraction):
    """Hegselmann-Krause coupling, the move of every agent to the mean opinion of the neighbours within its confidence.

    Every step every agent takes at once the mean opinion of its neighbours, itself included, whose opinions differ
    from its own by less than its parameter `confidence`. It is the map form of coupling with the normalized mean over
    the neighbours of the network weighted by the indicator of the confidence.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of agents, on the nodes of a network.

        state:
            The name of the state that holds the opinions.

    Examples:
        >>> import networkx as nx
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import LatticeElement
        >>> from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.social.opinion_dynamics import ConfidenceCouplingInteraction, ConfidentElement
        >>> people = LatticeElement(
        ...     'people',
        ...     element=ConfidentElement,
        ...     space=NetworkSpace(nx.complete_graph(3)),
        ...     values={'opinion': [0.1, 0.2, 0.9], 'confidence': 0.2},
        ... )
        >>> coupling = ConfidenceCouplingInteraction('compromise', target='people')
        >>> system = System(elements=[people], interactions=[coupling], time=DiscreteTime())
        >>> opinions = Simulation(system).prepare().run(until=1).trajectory_.values('people.opinion')
        >>> np.round(opinions[-1], 6).tolist()
        [0.15, 0.15, 0.9]
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], state: str = 'opinion') -> None:
        super().__init__(
            label,
            target=target,
            read=state,
            write=state,
            function=self._get_opinion,
            reduce='normalized-mean',
            weights=self._get_trust,
            form='map',
            neighborhood=AdjacencyNeighborhood(center=True),
        )
        self.state = state

    def _get_opinion(self, namespace: SimpleNamespace) -> Expression:
        """Return the opinion of the neighbour."""
        return getattr(namespace.other, self.state)

    def _get_trust(self, namespace: SimpleNamespace) -> Expression:
        """Return 1 for a neighbour whose opinion lies within the confidence of the agent, and 0 otherwise."""
        difference = getattr(namespace.other, self.state) - getattr(namespace.member, self.state)
        return where(abs(difference) < namespace.member.confidence, 1, 0)


class EncountersInteraction(RandomPairsInteraction):
    """Random binary encounters of the Deffuant system, as many per step as there are agents.

    In every encounter two random agents whose opinions differ by less than the confidence bound move towards each
    other by the convergence fraction of their difference.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of discussants.

        phase:
            The phase of the step the encounters run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import EncountersInteraction
        >>> EncountersInteraction('meetings', target='people').target
        'people'
    """

    def act_on(self, members: list[AgentViewProtocol], system: SystemViewProtocol) -> None:
        """Move the opinions of a pair towards each other when they differ by less than the confidence bound.

        Args:
            members:
                The two discussants of the encounter.

            system:
                The system view of the rule.
        """
        first, second = [member for member in members if isinstance(member, DiscussantView)]
        difference = second.opinion - first.opinion
        if abs(difference) < first.confidence:
            first.opinion += first.convergence * difference
            second.opinion -= first.convergence * difference


PAIR = 2


@register_system(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'bounded confidence', 'clusters'),
    references=('Deffuant et al. 2000', 'Castellano, Fortunato, and Loreto 2009', 'Gros 2015'),
    defaults={'until': 50},
)
def deffuant(
    n_agents: int = 200,
    *,
    confidence: float = 0.2,
    convergence: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Deffuant bounded confidence system, whose opinions settle into about `1 / (2 * confidence)` clusters.

    Args:
        n_agents:
            The number of agents.

        confidence:
            The largest opinion difference at which two agents compromise.

        convergence:
            The fraction of the difference by which each agent of a compromising pair moves.

        random_state:
            The seed or generator of the uniform initial opinions.

    Returns:
        system:
            The system, with the group `people`.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import deffuant
        >>> deffuant(20, random_state=0).elements[0].prepare().size_
        20
    """
    opinions = np.random.default_rng(random_state).random(n_agents)
    visual = VisualSpec('histograms', {'color_by': 'opinion', 'color_range': (0.0, 1.0), 'histogram_first': True})
    people = GroupElement(
        'people',
        element=DiscussantElement,
        size=n_agents,
        values={'opinion': opinions, 'confidence': confidence, 'convergence': convergence},
        visual=visual,
    )
    return System(elements=[people], interactions=[EncountersInteraction('meetings', target='people')])


@register_system(
    field='social/opinion_dynamics',
    tags=('opinion', 'bounded confidence', 'clusters', 'network', 'synchronous'),
    references=('Hegselmann and Krause 2002', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 30},
)
def hegselmann_krause(
    n_agents: int = 200,
    *,
    confidence: float = 0.15,
    graph: nx.Graph | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Hegselmann-Krause system, whose opinions freeze within a few steps into clusters about `2 ε` apart.

    Every step every agent moves at once to the mean opinion of its neighbours, itself included, whose opinions lie
    within the confidence `ε` of its own. On the complete graph the opinions settle into about `1 / (2 ε)` clusters.

    Args:
        n_agents:
            The number of agents, the nodes of the complete graph when no graph is given.

        confidence:
            The confidence `ε`.

        graph:
            The network of the agents, or `None` for the complete graph.

        random_state:
            The seed or generator of the uniform initial opinions.

    Returns:
        system:
            The system, with the group `people` on the nodes of the network.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.social.opinion_dynamics import hegselmann_krause
        >>> trajectory = Simulation(
        ...     hegselmann_krause(50, confidence=0.3), random_state=0,
        ... ).prepare().run(until=20).trajectory_
        >>> len(np.unique(np.round(trajectory.values('people.opinion')[-1], 6))) <= 3
        True
    """
    space = NetworkSpace(graph if graph is not None else nx.complete_graph(n_agents)).prepare()
    opinions = np.random.default_rng(random_state).random(space.size_)
    people = LatticeElement(
        'people',
        element=ConfidentElement,
        space=space,
        values={'opinion': opinions, 'confidence': confidence},
    )
    coupling = ConfidenceCouplingInteraction('compromise', target='people')
    return System(elements=[people], interactions=[coupling], time=DiscreteTime())
