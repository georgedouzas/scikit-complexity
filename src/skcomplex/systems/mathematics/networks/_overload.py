"""Cascades of overload failures.

The load of a node is the number of shortest paths through it.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction, TopologyInteraction
from ....core.components.laws.utils import components, shortest_paths
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import StateSpec, VisualSpec


class LoadedNodeElement(Element):
    """Node of a network that carries a load and fails above its capacity.

    The load is the number of shortest paths through it.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        load:
            The betweenness of the node on the network present, read at the start of every step.

        capacity:
            The largest load the node carries, a negative value for a node that fails at the first step.

        giant:
            The number of nodes of the largest connected component, the same for every node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import LoadedNodeElement
        >>> LoadedNodeElement('nodes', capacity=12.0).capacity
        12.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        load: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        capacity: Annotated[float, StateSpec()] = 0.0,
        giant: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.load = load
        self.capacity = capacity
        self.giant = giant


ATTACKS = ('load', 'degree', 'random')


class MotterLaiOverloadSystem(BaseSystem):
    """The cascade of overload failures of Motter and Lai after one node is removed.

    The load of a node is its betweenness, the number of shortest paths between other pairs of nodes that pass through
    it, and its capacity is `(1 + tolerance)` times its load on the intact network (Motter and Lai 2002, as Artime et
    al. 2025, section 5, review it). The attack removes one node at the first step, the node of the largest load, of the
    largest degree, or a random node. Every later step the loads are those of the network present, and every node whose
    load exceeds its capacity fails at once, until no node is overloaded. A heterogeneous network loses most of its
    largest component after an attack on a node of large load and a low tolerance, and keeps it after a random failure.
    The system has the nodes `nodes`, their `load`, `capacity`, and the size of the largest component `giant`, and the
    links `links`.

    Args:
        graph:
            The NetworkX graph, or `None` for a Barabási-Albert graph of 300 nodes with two edges per new node.

        tolerance:
            The spare capacity `alpha` of every node, as a share of its initial load.

        attack:
            `'load'` for the node of the largest load, `'degree'` for that of the largest degree, or `'random'`.

        random_state:
            The seed or generator of the graph and of the random attack.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The attack is unknown.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import MotterLaiOverloadSystem
        >>> trajectory = simulate(MotterLaiOverloadSystem(tolerance=0.1), until=10)
        >>> int(trajectory.network(-1).nodes.size) < 300
        True
    """

    field = 'mathematics/networks'
    topics = ('network', 'robustness', 'cascade', 'betweenness', 'overload')
    references = ('Motter and Lai 2002', 'Artime et al. 2025')
    until = 20

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        tolerance: float = 0.2,
        attack: str = 'load',
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.tolerance = tolerance
        self.attack = attack
        self.random_state = random_state

    @staticmethod
    def _is_overloaded(s: SimpleNamespace) -> object:
        """Return whether the load of a node on the network present exceeds its capacity."""
        return shortest_paths(1) > s.source.capacity

    def _build_system(self) -> System:
        """Return the cascade of overload failures of Motter and Lai.

        The cascade follows the removal of one node.
        """
        if self.attack not in ATTACKS:
            error_msg = f'Unknown attack {self.attack!r}. Use one of {", ".join(ATTACKS)}.'
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        graph = nx.barabasi_albert_graph(300, 2, seed=int(rng.integers(2**31))) if self.graph is None else self.graph
        graph = nx.convert_node_labels_to_integers(graph)
        found = nx.betweenness_centrality(graph, normalized=False)
        loads = np.asarray([found[node] for node in graph.nodes], dtype=float)
        degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
        target = {
            'load': int(np.argmax(loads)),
            'degree': int(np.argmax(degrees)),
            'random': int(rng.integers(len(loads))),
        }[self.attack]
        capacities = (1 + self.tolerance) * loads
        capacities[target] = -1.0
        nodes = GroupElement(
            'nodes',
            element=LoadedNodeElement,
            on='sites',
            variable=True,
            capacity=len(loads),
            values={'load': loads, 'capacity': capacities, 'giant': float(len(loads))},
        )
        links = GroupElement('links', element=Element, on='edges', variable=True, capacity=graph.number_of_edges())
        return System(
            self.label,
            elements=[nodes, links],
            interactions=[
                MapInteraction('load', target='nodes', state='load', function=lambda s: shortest_paths(1)),
                MapInteraction(
                    'giant',
                    target='nodes',
                    state='giant',
                    function=lambda s: components(1, reduce='largest'),
                ),
                TopologyInteraction('overload', target='nodes', action='remove-node', condition=self._is_overloaded),
            ],
            space=NetworkSpace(graph),
            time=DiscreteTime(),
        )
