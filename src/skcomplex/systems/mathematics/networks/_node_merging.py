"""Node-merging network of Kim et al.: linked nodes merge and newcomers arrive, which builds hubs at a fixed size."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction, TopologyInteraction
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.specs import StateSpec, VisualSpec
from ..._catalog import register_system

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})


class MergingNodeElement(Element):
    """Node of the node-merging network, with its degree.

    A map of the system reads the degree from the edges at the start of every step.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        degree:
            The number of edges of the node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import MergingNodeElement
        >>> MergingNodeElement('nodes').degree
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = DEGREE_VISUAL,
        *,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.degree = degree


def _system(capacity: int, space: NetworkSpace, links: int, changes: list[TopologyInteraction]) -> System:
    """Return the network of variable groups on its nodes and links, its changes, and the map of the degrees."""
    nodes = GroupElement('nodes', element=MergingNodeElement, on='sites', variable=True, capacity=capacity)
    edges = GroupElement('links', element=Element, on='edges', variable=True, capacity=links)
    degree = MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1))
    return System(elements=[nodes, edges], interactions=[*changes, degree], space=space, time=DiscreteTime())


@register_system(
    field='mathematics/networks',
    tags=('network', 'merging', 'edge contraction', 'scale-free', 'non-growing network'),
    references=('Kim, Trusina, Minnhagen, and Sneppen 2005', 'Thurner et al. 2018'),
    defaults={'until': 10000},
)
def node_merging(
    n_nodes: int = 500,
    *,
    edges: int = 1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the node-merging network of Kim et al., whose merging of linked nodes builds hubs without growth.

    The network starts as a random tree of `n_nodes` nodes. Every step a node drawn uniformly merges with one of its
    neighbours drawn uniformly into one node that keeps the union of their links, and a new node joins `edges` nodes
    drawn uniformly, so the network keeps its size (Thurner et al. 2018, section 4.5.4.6). Merging adds the links of
    the neighbour to the node, so well-linked nodes absorb more links, and the degrees follow a fat-tailed,
    q-exponential distribution with hubs far above the degrees of a random graph.

    Args:
        n_nodes:
            The number of nodes.

        edges:
            The number of links of every new node.

        random_state:
            The seed or generator of the initial tree.

    Returns:
        system:
            The system, with the groups `nodes` and `links`, the merging `merging`, and the arrival `arrival`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import node_merging
        >>> trajectory = Simulation(node_merging(50), random_state=0).prepare().run(until=20).trajectory_
        >>> network = trajectory.network(-1)
        >>> len(network.nodes), len(network.sources)
        (50, 49)
    """
    graph = nx.random_labeled_tree(n_nodes, seed=np.random.default_rng(random_state))
    merging = TopologyInteraction('merging', target='nodes', action='contract-edge', one=True)
    arrival = TopologyInteraction('arrival', action='add-node', edges=edges)
    capacity = 4 * edges * n_nodes
    return _system(n_nodes, NetworkSpace(graph), capacity, [merging, arrival])
