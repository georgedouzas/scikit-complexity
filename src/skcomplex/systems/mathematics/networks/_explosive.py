"""Explosive percolation.

Edges chosen by the Achlioptas product rule delay and sharpen the giant component.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction, TopologyInteraction
from ....core.components.laws.utils import components
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, StateSpec, VisualSpec

SIZE_VISUAL = VisualSpec('sites', {'states': ['size']})


class ClusterNodeElement(Element):
    """Node of a network with the number of nodes of its connected component.

    A map of the system reads the size from the edges at the start of every step, so it holds the size of the
    component in the network the requests of the step before left.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        size:
            The number of nodes of the connected component of the node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import ClusterNodeElement
        >>> ClusterNodeElement('nodes').size
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = SIZE_VISUAL,
        *,
        size: Annotated[float, StateSpec(domain=RealSet.closed_open(1, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.size = size


class ExplosivePercolationSystem(BaseSystem):
    """Explosive percolation.

    The Achlioptas product rule delays and sharpens the giant component.

    The network starts with `n_nodes` isolated nodes, and every step adds one edge. It draws `candidates` edges, each
    between two distinct nodes not yet joined drawn uniformly, and adds the one whose product of the sizes of the
    components it joins is smallest, the product rule of Achlioptas, D'Souza, and Spencer (2009). A map reads the size
    of the component of every node at the start of every step. With one candidate the edges are those of an Erdős-Rényi
    graph, whose giant component grows continuously from `t = 1/2` edges per node. With two candidates the rule
    suppresses large components until the giant component jumps in a narrow window near `t = 0.888`. The system has the
    size of the component of every node `nodes.size`.

    Args:
        n_nodes:
            The number of nodes.

        candidates:
            The number of candidate edges of every step.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import ExplosivePercolationSystem
        >>> trajectory = simulate(ExplosivePercolationSystem(100), until=50, random_state=0)
        >>> len(trajectory.network(-1).sources)
        50
    """

    field = 'mathematics/networks'
    topics = ('network', 'percolation', 'explosive percolation', 'Achlioptas process', 'giant component')
    references = ("Achlioptas, D'Souza, and Spencer 2009", 'Artime et al. 2025')
    until = 1000
    observables = (ObservableSpec('largest', group='nodes', variable='size', reduce='max'),)

    def __init__(
        self,
        n_nodes: int = 1000,
        *,
        candidates: int = 2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.candidates = candidates

    @staticmethod
    def _get_product(node: SimpleNamespace) -> object:
        """Return the score of the product rule of a candidate edge.

        The score is the product of the sizes of the components the edge joins.
        """
        return node.size * node.source.size

    def _build_system(self) -> CompositeSystem:
        """Return explosive percolation.

        The Achlioptas product rule delays and sharpens the giant component.
        """
        nodes = GroupElement('nodes', element=ClusterNodeElement, on='sites', variable=True, capacity=self.n_nodes)
        links = GroupElement('links', element=Element, on='edges', variable=True, capacity=2 * self.n_nodes)
        interactions = [
            MapInteraction('size', target='nodes', state='size', function=lambda s: components(1, reduce='size')),
            TopologyInteraction('linking', action='add-edge', candidates=self.candidates, score=self._get_product),
        ]
        space = NetworkSpace(nx.empty_graph(self.n_nodes))
        return CompositeSystem(
            self.label,
            elements=[nodes, links],
            interactions=interactions,
            space=space,
            time=DiscreteTime(),
        )
