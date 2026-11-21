"""Cascades of overload failures.

The load of a node is the number of shortest paths through it.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import components, shortest_paths, where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role

ATTACKS = ('load', 'degree', 'random')


class LoadedNodeElement(Element):
    """Nodes of a network, each with its load, its capacity, and the size of the largest component.

    Args:
        label:
            The label of the nodes.

        load:
            The initial betweenness of every node, one per node or shared.

        capacity:
            The largest load a node carries, a negative value for a node that fails at the first step, one per node or
            shared.

        giant:
            The initial number of nodes of the largest connected component, the same for every node.

        space:
            The network with one node per site, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import LoadedNodeElement
        >>> LoadedNodeElement('nodes', capacity=12.0).capacity
        12.0
    """

    def __init__(
        self,
        label: str,
        *,
        load: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        capacity: object = 0.0,
        giant: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.load = load
        self.capacity = capacity
        self.giant = giant


class OverloadInteraction(Interaction):
    """Failure of every node whose load, the number of shortest paths through it, exceeds its capacity.

    At the start of every step the load of every node and the size of the largest component are read from the network
    present, and every overloaded node leaves the network at the end of the step.

    Args:
        label:
            The label of the interaction.

        target:
            The nodes whose loads and sizes the interaction sets and that fail.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import OverloadInteraction
        >>> OverloadInteraction('overload').target
        'nodes'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'nodes') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the loads, the size of the largest component, and the failures as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        node = context.target
        self.laws_: list[BaseLaw] = [
            UpdateLaw(node.load, shortest_paths(1)),
            UpdateLaw(node.giant, components(1, reduce='largest')),
            UpdateLaw(TopologyRequestSpec('remove-node'), where(shortest_paths(1) > node.capacity, 1, 0)),
        ]


class MotterLaiOverloadSystem(BaseSystem):
    """The cascade of overload failures of Motter and Lai after one node is removed.

    The load of a node is its betweenness, the number of shortest paths between other pairs of nodes that pass through
    it, and its capacity is `(1 + tolerance)` times its load on the intact network (Motter and Lai 2002, as Artime et
    al. 2025, section 5, review it). The attack removes one node at the first step, the node of the largest load, of the
    largest degree, or a random node. Every later step the loads are those of the network present, and every node whose
    load exceeds its capacity fails at once, until no node is overloaded. A heterogeneous network loses most of its
    largest component after an attack on a node of large load and a low tolerance, and keeps it after a random failure.
    The system has the nodes `nodes`, their `load`, `capacity`, and the size of the largest component `giant`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The attack is unknown.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import MotterLaiOverloadSystem
        >>> trajectory = MotterLaiOverloadSystem(tolerance=0.1).prepare().simulate(until=10)
        >>> int(trajectory.get_network(-1).nodes.size) < 300
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.tolerance = tolerance
        self.attack = attack
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the loaded nodes of the network after the attack.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The attack is unknown.
        """
        super()._prepare(context)
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
        nodes = LoadedNodeElement(
            'nodes',
            load=loads,
            capacity=capacities,
            giant=float(len(loads)),
            space=NetworkSpace(graph),
        )
        self.system_ = System(self.label, elements=[nodes], interactions=[OverloadInteraction('overload')])
