"""Node-merging network of Kim and coauthors.

Linked nodes merge and newcomers arrive, which builds hubs at a fixed size.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})


class MergingNodeElement(Element):
    """Node of the node-merging network, with its degree.

    A map of the system reads the degree from the edges at the start of every step.

    Args:
        label:
            The label of the node group.

        degree:
            The number of edges of the node.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import MergingNodeElement
        >>> MergingNodeElement('nodes').degree
        0.0
    """

    visual = DEGREE_VISUAL

    def __init__(
        self,
        label: str,
        *,
        degree: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.degree = degree


class NodeMergingSystem(BaseSystem):
    """The node-merging network of Kim and coauthors.

    Its merging of linked nodes builds hubs without growth.

    The network starts as a random tree of `n_nodes` nodes. Every step a node drawn uniformly merges with one of its
    neighbours drawn uniformly into one node that keeps the union of their links, and a new node joins `edges` nodes
    drawn uniformly, so the network keeps its size (Thurner et al. 2018, section 4.5.4.6). Merging adds the links of the
    neighbour to the node, so well-linked nodes absorb more links, and the degrees follow a fat-tailed, q-exponential
    distribution with hubs far above the degrees of a random graph. The system has the groups `nodes` and `links`, the
    merging `merging`, and the arrival `arrival`.

    Args:
        n_nodes:
            The number of nodes.

        edges:
            The number of links of every new node.

        random_state:
            The seed or generator of the initial tree.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NodeMergingSystem
        >>> trajectory = NodeMergingSystem(50).prepare().simulate(until=20, random_state=0)
        >>> network = trajectory.get_network(-1)
        >>> len(network.nodes), len(network.sources)
        (50, 49)
    """

    field = 'mathematics/networks'
    topics = ('network', 'merging', 'edge contraction', 'scale-free', 'non-growing network')
    references = ('Kim, Trusina, Minnhagen, and Sneppen 2005', 'Thurner et al. 2018')
    until = 10000

    def __init__(
        self,
        n_nodes: int = 500,
        *,
        edges: int = 1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.edges = edges
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the node-merging network of Kim and coauthors, whose merging builds hubs."""
        super()._prepare(context)
        graph = nx.random_labeled_tree(self.n_nodes, seed=np.random.default_rng(self.random_state))
        merging = Interaction(
            'merging',
            roles={'target': 'nodes'},
            laws=lambda s: [UpdateLaw(TopologyRequestSpec('contract-edge', one=True), 1)],
        )
        arrival = Interaction('arrival', laws=[UpdateLaw(TopologyRequestSpec('add-node', edges=self.edges), 1)])
        space = NetworkSpace(graph)
        nodes = MergingNodeElement('nodes', space=space)
        edges = Element('links', space=space, on='edges')
        degree = Interaction(
            'degree',
            roles={'target': 'nodes'},
            laws=lambda s: [UpdateLaw(s.target.degree, neighbors(1))],
        )
        self.system_ = System(
            self.label,
            elements=[nodes, edges],
            interactions=[merging, arrival, degree],
            space=space,
        )
