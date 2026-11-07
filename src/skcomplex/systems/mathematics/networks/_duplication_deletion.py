"""Duplication-deletion network of Jensen: nodes die and duplicate, and the connectance settles at a steady value."""

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


class DuplicatingNodeElement(Element):
    """Node of the duplication-deletion network, with its degree.

    A map of the system reads the degree from the edges at the start of every step.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        degree:
            The number of edges of the node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import DuplicatingNodeElement
        >>> DuplicatingNodeElement('nodes').degree
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
    nodes = GroupElement('nodes', element=DuplicatingNodeElement, on='sites', variable=True, capacity=capacity)
    edges = GroupElement('links', element=Element, on='edges', variable=True, capacity=links)
    degree = MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1))
    return System(elements=[nodes, edges], interactions=[*changes, degree], space=space, time=DiscreteTime())


@register_system(
    field='mathematics/networks',
    tags=('network', 'duplication', 'deletion', 'evolution', 'protein interaction network', 'connectance'),
    references=('Jensen 2023', 'Sole et al. 2002'),
    defaults={'until': 4000},
)
def duplication_deletion(
    n_nodes: int = 100,
    *,
    kill: float = 1.0,
    inherit: float = 0.5,
    parent: float = 0.5,
    new: float = 0.01,
    density: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the steady-state duplication-deletion network of Jensen, whose connectance settles at Eq. 8.84.

    The network starts as an Erdős-Rényi graph of `n_nodes` nodes and the link probability `density`. Every step a node
    drawn uniformly leaves with its links with probability `kill`, and a parent drawn uniformly among the nodes left is
    duplicated: the offspring inherits each link of the parent with probability `inherit`, joins the parent with
    probability `parent`, and joins every node the parent does not join with probability `new` (Jensen 2023, section
    8.5.1). With `kill = 1` the network keeps `N` nodes, and its connectance converges to
    `C = (new (N - 2) + parent) / (N - 1 + (new - inherit) (N - 2))`, Eq. 8.84. Accurate duplication, with `inherit`
    near 1 and `new` near 0, turns the binomial degrees of noisy duplication into a cut-off power law. With `kill < 1`
    the network grows by `1 - kill` nodes per step on average, up to twice its initial size.

    Args:
        n_nodes:
            The number of nodes.

        kill:
            The probability that the drawn node leaves at a step.

        inherit:
            The probability that the offspring inherits a link of its parent.

        parent:
            The probability that the offspring joins its parent.

        new:
            The probability that the offspring joins a node its parent does not join.

        density:
            The link probability of the initial Erdős-Rényi graph.

        random_state:
            The seed or generator of the initial graph.

    Returns:
        system:
            The system, with the groups `nodes` and `links`, the deletion `death`, and the duplication `birth`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import duplication_deletion
        >>> trajectory = Simulation(duplication_deletion(30), random_state=0).prepare().run(until=20).trajectory_
        >>> len(trajectory.network(-1).nodes)
        30
    """
    graph = nx.gnp_random_graph(n_nodes, density, seed=np.random.default_rng(random_state))
    death = TopologyInteraction('death', target='nodes', action='remove-node', one=True, probability=kill)
    birth = TopologyInteraction(
        'birth',
        target='nodes',
        action='duplicate-node',
        one=True,
        inherit=inherit,
        parent=parent,
        independent=True,
        toward=lambda node: new,
    )
    capacity = n_nodes * (2 * n_nodes - 1)
    return _system(2 * n_nodes, NetworkSpace(graph), capacity, [death, birth])
