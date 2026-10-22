"""Cognitive consistency models: attitudes that spread between agents and settle inside each agent's beliefs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.model import Element, Interaction, Law, NetworkSpace, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import ModelView

type Array = npt.NDArray[np.float64]

TOPICS = (0, 2)
POSITIVE = (1.0, 1.0, -1.0)
COHERENCE = np.array([[0.0, 0.05, 0.0], [0.05, 0.0, -0.05], [0.0, -0.05, 0.0]])
SETTLED = 0.01
MAX_SETTLING = 1000


def _settle_once(valences: Array, decay: float) -> Array:
    """Return the valences after one synchronous update of the parallel constraint satisfaction network."""
    net = COHERENCE @ valences
    growth = np.where(net > 0, net * (1.0 - valences), net * (valences + 1.0))
    return valences * (1.0 - decay) + growth


def settle(valences: npt.ArrayLike, decay: float = 0.05) -> Array:
    """Return the valences of a belief network updated until no valence changes by more than 0.01.

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
        >>> from skcomplex.models.social.opinion_dynamics import settle
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


class Thinker(Element):
    """Agent of the cognitive consistency model, with the valences of the concepts of its belief network.

    Args:
        label:
            The label of the agent or group.

        valences:
            The initial valences of the concepts F1, X, and F2, between -1 and 1.

        decay:
            The fraction of every valence lost per update of the belief network.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Thinker
        >>> Thinker.group('thinkers', 5, valences=[1.0, 1.0, -1.0])
        Group('thinkers', element=Thinker, size=5)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'attitude', 'color_range': (-1.0, 1.0)})
    valences: npt.ArrayLike = state(default=0.0, vector=True)
    attitude: float = state(default=0.0)
    decay: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)


class Conversations(Interaction):
    """Conversations of the cognitive consistency model, as many per step as there are agents.

    A random speaker talks to a random neighbour about F1 or F2. The hearer averages its valence of the topic with the
    speaker's and settles its belief network, and its attitude is its valence of X.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of thinkers.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Conversations
        >>> Conversations('talks', target='thinkers').target
        'thinkers'
    """

    target: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the conversation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.converse)]

    def converse(self, model: ModelView) -> None:
        """Run one conversation per agent.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        for _ in range(len(agents)):
            speaker = agents[int(model.rng.integers(len(agents)))]
            neighbors = speaker.neighbors()
            if not neighbors:
                continue
            hearer = neighbors[int(model.rng.integers(len(neighbors)))]
            topic = TOPICS[int(model.rng.integers(len(TOPICS)))]
            valences = np.array(hearer.valences, dtype=float)
            valences[topic] = (valences[topic] + np.asarray(speaker.valences, dtype=float)[topic]) / 2
            settled = settle(valences, hearer.decay)
            hearer.valences, hearer.attitude = settled, float(settled[1])


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'network', 'cognition'),
    references=('Lakkaraju and Speed 2019', 'Carmichael et al. 2019'),
    defaults={'until': 30},
)
def cognitive_consistency(
    graph: object = None,
    *,
    positive: float = 0.7,
    decay: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return agents whose attitudes spread by conversation and settle in a belief network that seeks consistency.

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

    Returns:
        system:
            The model, with the group `thinkers`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import cognitive_consistency
        >>> cognitive_consistency(nx.complete_graph(6), random_state=0).elements[0].size
        6
    """
    graph = nx.complete_graph(40) if graph is None else graph
    size = nx.number_of_nodes(graph)
    signs = np.where(np.random.default_rng(random_state).random(size) < positive, 1.0, -1.0)
    valences = signs[:, None] * np.array(POSITIVE)
    thinkers = Thinker.group('thinkers', size, valences=valences, attitude=valences[:, 1], decay=decay)
    return System(
        components=[thinkers],
        interactions=[Conversations('talks', target='thinkers')],
        space=NetworkSpace(graph),
    )
