"""Explosive percolation.

Edges chosen by the Achlioptas product rule delay and sharpen the giant component.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction, TopologyInteraction
from ....core.components.laws.utils import components
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System

SIZE_VISUAL = VisualSpec('sites', {'states': ['cluster_size']})


class ClusterNodeElement(Element):
    """Node of a network with the number of nodes of its connected component.

    A map of the system reads the size from the edges at the start of every step, so it holds the size of the
    component in the network the requests of the step before left.

    Args:
        label:
            The label of the node group.

        cluster_size:
            The number of nodes of the connected component of the node.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import ClusterNodeElement
        >>> ClusterNodeElement('nodes').cluster_size
        1.0
    """

    visual = SIZE_VISUAL

    def __init__(
        self,
        label: str,
        *,
        cluster_size: Annotated[float, RealSet.closed_open(1, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.cluster_size = cluster_size


class ExplosivePercolationSystem(BaseSystem):
    """Explosive percolation.

    The Achlioptas product rule delays and sharpens the giant component.

    The network starts with `n_nodes` isolated nodes, and every step adds one edge. It draws `candidates` edges, each
    between two distinct nodes not yet joined drawn uniformly, and adds the one whose product of the sizes of the
    components it joins is smallest, the product rule of Achlioptas, D'Souza, and Spencer (2009). A map reads the size
    of the component of every node at the start of every step. With one candidate the edges are those of an Erdős-Rényi
    graph, whose giant component grows continuously from `t = 1/2` edges per node. With two candidates the rule
    suppresses large components until the giant component jumps in a narrow window near `t = 0.888`. The system has the
    size of the component of every node `nodes.cluster_size`.

    Args:
        n_nodes:
            The number of nodes.

        candidates:
            The number of candidate edges of every step.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import ExplosivePercolationSystem
        >>> trajectory = ExplosivePercolationSystem(100).prepare().simulate(until=50, random_state=0)
        >>> len(trajectory.get_network(-1).sources)
        50
    """

    field = 'mathematics/networks'
    topics = ('network', 'percolation', 'explosive percolation', 'Achlioptas process', 'giant component')
    references = ("Achlioptas, D'Souza, and Spencer 2009", 'Artime et al. 2025')
    until = 1000
    observables = (Observable('largest', group='nodes', variable='cluster_size', reduce='max'),)

    def __init__(
        self,
        n_nodes: int = 1000,
        *,
        candidates: int = 2,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.candidates = candidates

    @staticmethod
    def _get_product(node: SimpleNamespace) -> object:
        """Return the score of the product rule of a candidate edge.

        The score is the product of the sizes of the components the edge joins.
        """
        return node.cluster_size * node.source.cluster_size

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to explosive percolation, whose product rule delays and sharpens the giant component."""
        super()._prepare(context)
        space = NetworkSpace(nx.empty_graph(self.n_nodes))
        nodes = ClusterNodeElement('nodes', space=space)
        links = Element('links', space=space, on='edges')
        interactions = [
            MapInteraction(
                'size',
                target='nodes',
                state='cluster_size',
                function=lambda s: components(1, reduce='size'),
            ),
            TopologyInteraction('linking', action='add-edge', candidates=self.candidates, score=self._get_product),
        ]
        self.system_ = System(self.label, elements=[nodes, links], interactions=interactions, space=space)
