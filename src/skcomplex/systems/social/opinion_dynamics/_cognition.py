"""Cognitive consistency systems: attitudes that spread between agents.

The attitudes settle inside each agent's beliefs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.spaces import NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver, SystemView
from ....core.specs import Role, VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ._random_pairs import NeighborhoodsInteraction

TOPICS = (0, 2)
COHERENCE = np.array([[0.0, 0.05, 0.0], [0.05, 0.0, -0.05], [0.0, -0.05, 0.0]])
SETTLED = 0.01
MAX_SETTLING = 1000
POSITIVE = (1.0, 1.0, -1.0)


class ThinkerView(AgentView):
    """View of an agent of the cognitive consistency system.

    Its states and parameters are attributes.
    """

    valences: Array
    attitude: float
    decay: float


class ThinkerElement(Element):
    """Agent of the cognitive consistency system.

    It holds the valences of the concepts of its belief network.

    Args:
        label:
            The label of the agent or group.

        valences:
            The initial valences of the concepts F1, X, and F2, between -1 and 1.

        attitude:
            The attitude of the agent, the valence of the concept X once the belief network settles.

        decay:
            The fraction of every valence lost per update of the belief network.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import ThinkerElement
        >>> group = ThinkerElement('thinkers', valences=[1.0, 1.0, -1.0], size=5)
        >>> group.label, type(group).__name__, group.size
        ('thinkers', 'ThinkerElement', 5)
    """

    visual = VisualSpec('agents', {'color_by': 'attitude', 'color_range': (-1.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        valences: npt.ArrayLike = 0.0,
        attitude: object = 0.0,
        decay: Annotated[object, RealSet.closed_open(0, oo)] = 0.05,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.valences = valences
        self.attitude = attitude
        self.decay = decay

    @classmethod
    def _get_view(cls) -> type[ThinkerView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return ThinkerView


def _settle_once(valences: Array, decay: float) -> Array:
    """Return the valences after one synchronous update.

    The update is that of the parallel constraint satisfaction network.
    """
    net = COHERENCE @ valences
    growth = np.where(net > 0, net * (1.0 - valences), net * (valences + 1.0))
    return valences * (1.0 - decay) + growth


def settle(valences: npt.ArrayLike, decay: float = 0.05) -> Array:
    """Return the valences of a belief network updated until they settle.

    The updates stop when no valence changes by more than 0.01.

    The network links the first concept to the second with weight 0.05 and the second to the third with weight
    -0.05. Every update decays a valence by `decay` and moves it towards 1 or -1 by its net input.

    Args:
        valences:
            The valences of the three concepts, between -1 and 1.

        decay:
            The fraction of every valence lost per update.

    Returns:
        valences:
            The settled valences.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import settle
        >>> settle([1.0, 1.0, -1.0]).round(2).tolist()
        [0.48, 0.57, -0.48]
    """
    current = np.asarray(valences, dtype=float)
    for _ in range(MAX_SETTLING):
        following = _settle_once(current, decay)
        if np.abs(following - current).max() < SETTLED:
            return following
        current = following
    return current


class ConversationsInteraction(NeighborhoodsInteraction):
    """Conversations of the cognitive consistency system.

    There is one per agent and step, the agents speaking in random order. A speaker talks to a random neighbour about F1
    or F2. The hearer averages its valence of the topic with the speaker's and settles its belief network, and its
    attitude is its valence of X.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of thinkers.

        phase:
            The phase of the step the conversations run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ConversationsInteraction
        >>> ConversationsInteraction('talks', target='thinkers').target
        'thinkers'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, phase=phase)

    def act_on(self, members: list[AgentView], system: SystemView) -> None:
        """Let the first member speak to a random neighbour.

        The neighbour averages the topic and settles its beliefs.

        Args:
            members:
                The speaker, followed by its neighbours.

            system:
                The system view of the rule.
        """
        speaker, *neighbors = cast(list[ThinkerView], members)
        if not neighbors:
            return
        hearer = neighbors[int(system.rng.integers(len(neighbors)))]
        topic = TOPICS[int(system.rng.integers(len(TOPICS)))]
        valences = np.array(hearer.valences, dtype=np.float64)
        valences[topic] = (valences[topic] + np.asarray(speaker.valences, dtype=np.float64)[topic]) / 2
        settled = settle(valences, hearer.decay)
        hearer.valences, hearer.attitude = settled, float(settled[1])


class CognitiveConsistencySystem(BaseSystem):
    """Agents whose attitudes spread by conversation and settle in a belief network.

    The belief network seeks consistency. The system has the group `thinkers`.

    Args:
        graph:
            The NetworkX graph, or `None` for a complete graph of 40 agents.

        positive:
            The share of agents that start in the positive state, F1 = X = 1 and F2 = -1, the others starting in the
            negative state.

        decay:
            The fraction of every valence lost per update of the belief network.

        random_state:
            The seed or generator of the initial states.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.opinion_dynamics import CognitiveConsistencySystem
        >>> system = CognitiveConsistencySystem(nx.complete_graph(6), random_state=0).prepare().system_
        >>> system.elements[0].size
        6
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'network', 'cognition')
    references = ('Lakkaraju and Speed 2019', 'Carmichael et al. 2019')
    until = 30

    def __init__(
        self,
        graph: object = None,
        *,
        positive: float = 0.7,
        decay: float = 0.05,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.positive = positive
        self.decay = decay
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        graph = self.graph
        graph = nx.complete_graph(40) if graph is None else graph
        size = nx.number_of_nodes(graph)
        signs = np.where(np.random.default_rng(self.random_state).random(size) < self.positive, 1.0, -1.0)
        valences = signs[:, None] * np.array(POSITIVE)
        thinkers = ThinkerElement('thinkers', valences=valences, attitude=valences[:, 1], decay=self.decay, size=size)
        self.system_ = System(
            self.label,
            elements=[thinkers],
            interactions=[ConversationsInteraction('talks', target='thinkers')],
            space=NetworkSpace(graph),
        )
