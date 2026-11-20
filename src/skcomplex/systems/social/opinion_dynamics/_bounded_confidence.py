"""Bounded confidence systems: compromise between agents whose opinions are close.

They compromise in pairs or all at once.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from scipy.special import expit

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import neighbors, where
from ....core.components.spaces import AdjacencyNeighborhood, BaseSpace, NetworkSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ._random_pairs import RandomPairsInteraction

PAIR = 2


class ConfidentElement(Element):
    """Agent of the Hegselmann-Krause system.

    It holds an opinion and the confidence within which it trusts neighbours. `ConfidenceCouplingInteraction` moves
    every agent each step to the mean opinion of the neighbours, itself included, whose opinions differ from its own by
    less than `confidence`.

    Args:
        label:
            The label of the agent or group.

        opinion:
            The opinion, between 0 and 1.

        confidence:
            The largest opinion difference of a neighbour an agent still listens to.

        space:
            The space with one member per site, or `None` for one member.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ConfidentElement
        >>> ConfidentElement('people', confidence=0.1).confidence
        0.1
    """

    visual = VisualSpec('histograms', {'color_range': (0.0, 1.0), 'histogram_first': True})

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        confidence: Annotated[object, RealSet.closed_open(0, oo)] = 0.2,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.opinion = opinion
        self.confidence = confidence


class DiscussantView(AgentViewProtocol, Protocol):
    """View of an agent of the Deffuant bounded confidence system.

    Its states and parameters are attributes.
    """

    opinion: float
    confidence: float
    convergence: float


class DiscussantElement(Element):
    """Agent of the Deffuant system, with a continuous opinion and a confidence bound.

    Args:
        label:
            The label of the agent or group.

        opinion:
            The initial opinion, between 0 and 1.

        confidence:
            The largest opinion difference at which two agents still compromise.

        convergence:
            The fraction of the opinion difference by which each agent of a compromising pair moves.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import DiscussantElement
        >>> group = DiscussantElement('people', opinion=0.5, size=10)
        >>> group.label, type(group).__name__, group.size
        ('people', 'DiscussantElement', 10)
    """

    visual = VisualSpec('histograms', {'color_by': 'opinion', 'color_range': (0.0, 1.0), 'histogram_first': True})

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        confidence: Annotated[object, RealSet.closed_open(0, oo)] = 0.2,
        convergence: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.opinion = opinion
        self.confidence = confidence
        self.convergence = convergence


class ConfidenceCouplingInteraction(Interaction):
    """Hegselmann-Krause coupling of the opinions of agents.

    Every step every agent takes at once the mean opinion of its neighbours, itself included, whose opinions differ from
    its own by less than its parameter `confidence`. It is the map form of coupling with the normalized mean over the
    neighbours of the network weighted by the indicator of the confidence.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of agents, on the nodes of a network.

        state:
            The name of the state that holds the opinions.

    Examples:
        >>> import networkx as nx
        >>> import numpy as np
        >>> from skcomplex.core.components.spaces import NetworkSpace
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.social.opinion_dynamics import ConfidenceCouplingInteraction, ConfidentElement
        >>> network = NetworkSpace(nx.complete_graph(3))
        >>> people = ConfidentElement('people', opinion=[0.1, 0.2, 0.9], confidence=0.2, space=network)
        >>> coupling = ConfidenceCouplingInteraction('compromise', target='people')
        >>> system = System(elements=[people], interactions=[coupling])
        >>> opinions = system.prepare().simulate(until=1).get_values('people.opinion')
        >>> np.round(opinions[-1], 6).tolist()
        [0.15, 0.15, 0.9]
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, state: str = 'opinion') -> None:
        super().__init__(label)
        self.target = target
        self.state = state

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next opinion, the mean of the trusted opinions of the agent and its neighbours, as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            member = context.target
            opinion, other = getattr(member, self.state), getattr(member.other, self.state)
            trust = where(abs(other - opinion) < member.confidence, 1, 0)
            mean = neighbors(other, over=AdjacencyNeighborhood(center=True), reduce='normalized-mean', weight=trust)
            self.laws_: list[BaseComponent] = [UpdateLaw(opinion, mean)]


class EncountersInteraction(RandomPairsInteraction):
    """Random binary encounters of the Deffuant system.

    There are as many per step as there are agents. In every encounter two random agents move towards each other by the
    convergence fraction of their difference when a Bernoulli outcome, drawn by the name `outcome`, is 1. Without a
    steepness the outcome is 1 when their opinions differ by less than the confidence bound. With a steepness `rho` it
    is 1 with the probability `1 / (1 + exp(-rho * (confidence - |x_u - x_v|)))` of Lenti et al. (2023), which
    approaches the threshold as `rho` grows. Runs record and replay the outcomes.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of discussants.

        phase:
            The phase of the step the encounters run in, or `None` for the default phase.

        steepness:
            The steepness `rho` of the probability of a compromise, or `None` for the deterministic threshold.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import EncountersInteraction
        >>> EncountersInteraction('meetings', target='people', steepness=20.0).steepness
        20.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        phase: str | None = None,
        steepness: float | None = None,
    ) -> None:
        super().__init__(label, target=target, phase=phase)
        self.steepness = steepness

    def act_on(self, members: list[AgentViewProtocol], system: SystemView) -> None:
        """Move the opinions of a pair towards each other when their drawn outcome is 1.

        Args:
            members:
                The two discussants of the encounter.

            system:
                The system view of the rule.
        """
        first, second = cast(list[DiscussantView], members)
        difference = second.opinion - first.opinion
        gap = first.confidence - abs(difference)
        probability = float(gap > 0) if self.steepness is None else float(expit(self.steepness * gap))
        if system.draw('outcome', 'bernoulli', p=probability):
            first.opinion += first.convergence * difference
            second.opinion -= first.convergence * difference


class DeffuantSystem(BaseSystem):
    """The Deffuant bounded confidence system.

    Its opinions settle into about `1 / (2 * confidence)` clusters. The system has the group `people`.

    Args:
        n_agents:
            The number of agents.

        confidence:
            The largest opinion difference at which two agents compromise.

        convergence:
            The fraction of the difference by which each agent of a compromising pair moves.

        steepness:
            The steepness of the probability that a pair compromises, after Lenti et al. (2023), or `None` for the
            deterministic threshold of the confidence.

        random_state:
            The seed or generator of the uniform initial opinions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import DeffuantSystem
        >>> DeffuantSystem(20, random_state=0).prepare().system_.elements[0].size
        20
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'bounded confidence', 'clusters')
    references = ('Deffuant et al. 2000', 'Castellano, Fortunato, and Loreto 2009', 'Gros 2015', 'Lenti et al. 2023')
    until = 50

    def __init__(
        self,
        n_agents: int = 200,
        *,
        confidence: float = 0.2,
        convergence: float = 0.5,
        steepness: float | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.confidence = confidence
        self.convergence = convergence
        self.steepness = steepness
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        opinions = np.random.default_rng(self.random_state).random(self.n_agents)
        people = DiscussantElement(
            'people',
            opinion=opinions,
            confidence=self.confidence,
            convergence=self.convergence,
            size=self.n_agents,
        )
        encounters = EncountersInteraction('meetings', target='people', steepness=self.steepness)
        self.system_ = System(self.label, elements=[people], interactions=[encounters])


class HegselmannKrauseSystem(BaseSystem):
    """The Hegselmann-Krause system.

    Its opinions freeze within a few steps into clusters about `2 ε` apart. Every step every agent moves at once to the
    mean opinion of its neighbours, itself included, whose opinions lie within the confidence `ε` of its own. On the
    complete graph the opinions settle into about `1 / (2 ε)` clusters. The system has the group `people` on the nodes
    of the network.

    Args:
        n_agents:
            The number of agents, the nodes of the complete graph when no graph is given.

        confidence:
            The confidence `ε`.

        graph:
            The network of the agents, or `None` for the complete graph.

        random_state:
            The seed or generator of the uniform initial opinions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.social.opinion_dynamics import HegselmannKrauseSystem
        >>> trajectory = HegselmannKrauseSystem(50, confidence=0.3).prepare().simulate(until=20, random_state=0)
        >>> len(np.unique(np.round(trajectory.get_values('people.opinion')[-1], 6))) <= 3
        True
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'bounded confidence', 'clusters', 'network', 'synchronous')
    references = ('Hegselmann and Krause 2002', 'Castellano, Fortunato, and Loreto 2009')
    until = 30

    def __init__(
        self,
        n_agents: int = 200,
        *,
        confidence: float = 0.15,
        graph: nx.Graph | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.confidence = confidence
        self.graph = graph
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        space = NetworkSpace(self.graph if self.graph is not None else nx.complete_graph(self.n_agents)).prepare()
        opinions = np.random.default_rng(self.random_state).random(space.size_)
        people = ConfidentElement('people', opinion=opinions, confidence=self.confidence, space=space)
        coupling = ConfidenceCouplingInteraction('compromise', target='people')
        self.system_ = System(self.label, elements=[people], interactions=[coupling])
