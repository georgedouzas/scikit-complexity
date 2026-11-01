"""Cognitive consistency models: attitudes that spread between agents and settle inside each agent's beliefs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import networkx as nx
import numpy as np

from ....core.elements import GroupElement
from ....core.spaces import NetworkSpace
from ....core.system import System
from ..._catalog import register_model
from ._conversations import ConversationsInteraction
from ._thinker import ThinkerElement

POSITIVE = (1.0, 1.0, -1.0)


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
        >>> cognitive_consistency(nx.complete_graph(6), random_state=0).elements[0].check().prepare().size_
        6
    """
    graph = nx.complete_graph(40) if graph is None else graph
    size = nx.number_of_nodes(graph)
    signs = np.where(np.random.default_rng(random_state).random(size) < positive, 1.0, -1.0)
    valences = signs[:, None] * np.array(POSITIVE)
    thinkers = GroupElement(
        'thinkers',
        element=ThinkerElement,
        size=size,
        values={'valences': valences, 'attitude': valences[:, 1], 'decay': decay},
    )
    return System(
        elements=[thinkers],
        interactions=[ConversationsInteraction('talks', target='thinkers')],
        space=NetworkSpace(graph),
    )
