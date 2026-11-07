"""Explosive percolation: edges chosen by the Achlioptas product rule, which delays and sharpens the giant component."""

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
from ....core.components.system import System
from ....core.specs import ObservableSpec, StateSpec, VisualSpec
from ..._catalog import register_system

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


def _largest_observables(system: System) -> list[ObservableSpec]:
    """Return the number of nodes of the largest component."""
    del system
    return [ObservableSpec('largest', group='nodes', variable='size', reduce='max')]


def _get_product(node: SimpleNamespace) -> object:
    """Return the product of the sizes of the components a candidate edge joins, the score of the product rule."""
    return node.size * node.source.size


@register_system(
    field='mathematics/networks',
    tags=('network', 'percolation', 'explosive percolation', 'Achlioptas process', 'giant component'),
    references=("Achlioptas, D'Souza, and Spencer 2009", 'Artime et al. 2025'),
    defaults={'until': 1000},
    observables=_largest_observables,
)
def explosive_percolation(n_nodes: int = 1000, *, candidates: int = 2) -> System:
    """Return explosive percolation, where the Achlioptas product rule delays and sharpens the giant component.

    The network starts with `n_nodes` isolated nodes, and every step adds one edge. It draws `candidates` edges, each
    between two distinct nodes not yet joined drawn uniformly, and adds the one whose product of the sizes of the
    components it joins is smallest, the product rule of Achlioptas, D'Souza, and Spencer (2009). A map reads the size
    of the component of every node at the start of every step. With one candidate the edges are those of an
    Erdős-Rényi graph, whose giant component grows continuously from `t = 1/2` edges per node. With two candidates the
    rule suppresses large components until the giant component jumps in a narrow window near `t = 0.888`.

    Args:
        n_nodes:
            The number of nodes.

        candidates:
            The number of candidate edges of every step.

    Returns:
        system:
            The system, with the size of the component of every node `nodes.size`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import explosive_percolation
        >>> trajectory = Simulation(explosive_percolation(100), random_state=0).prepare().run(until=50).trajectory_
        >>> len(trajectory.network(-1).sources)
        50
    """
    nodes = GroupElement('nodes', element=ClusterNodeElement, on='sites', variable=True, capacity=n_nodes)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=2 * n_nodes)
    interactions = [
        MapInteraction('size', target='nodes', state='size', function=lambda s: components(1, reduce='size')),
        TopologyInteraction('linking', action='add-edge', candidates=candidates, score=_get_product),
    ]
    space = NetworkSpace(nx.empty_graph(n_nodes))
    return System(elements=[nodes, links], interactions=interactions, space=space, time=DiscreteTime())
