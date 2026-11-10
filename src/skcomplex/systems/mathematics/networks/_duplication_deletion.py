"""Duplication-deletion network of Jensen.

Nodes die and duplicate, and the connectance settles at a steady value.
"""

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
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import StateSpec, VisualSpec

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


class DuplicationDeletionSystem(BaseSystem):
    """The steady-state duplication-deletion network of Jensen.

    Its connectance settles at Eq. 8.84.

    The network starts as an Erdős-Rényi graph of `n_nodes` nodes and the link probability `density`. Every step a node
    drawn uniformly leaves with its links with probability `kill`, and a parent drawn uniformly among the nodes left is
    duplicated: the offspring inherits each link of the parent with probability `inherit`, joins the parent with
    probability `parent`, and joins every node the parent does not join with probability `new` (Jensen 2023, section
    8.5.1). With `kill = 1` the network keeps `N` nodes, and its connectance converges to
    `C = (new (N - 2) + parent) / (N - 1 + (new - inherit) (N - 2))`, Eq. 8.84. Accurate duplication, with `inherit`
    near 1 and `new` near 0, turns the binomial degrees of noisy duplication into a cut-off power law. With `kill < 1`
    the network grows by `1 - kill` nodes per step on average, up to twice its initial size.

    The system has the groups `nodes` and `links`, the deletion `death`, and the duplication `birth`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import DuplicationDeletionSystem
        >>> trajectory = simulate(DuplicationDeletionSystem(30), until=20, random_state=0)
        >>> len(trajectory.get_network(-1).nodes)
        30
    """

    field = 'mathematics/networks'
    topics = ('network', 'duplication', 'deletion', 'evolution', 'protein interaction network', 'connectance')
    references = ('Jensen 2023', 'Sole et al. 2002')
    until = 4000

    def __init__(
        self,
        n_nodes: int = 100,
        *,
        kill: float = 1.0,
        inherit: float = 0.5,
        parent: float = 0.5,
        new: float = 0.01,
        density: float = 0.05,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.kill = kill
        self.inherit = inherit
        self.parent = parent
        self.new = new
        self.density = density
        self.random_state = random_state

    @staticmethod
    def _system(
        label: str,
        capacity: int,
        space: NetworkSpace,
        links: int,
        changes: list[TopologyInteraction],
    ) -> CompositeSystem:
        """Return the network of variable groups on its nodes and links.

        The network has its changes and the map of the degrees.
        """
        nodes = GroupElement('nodes', element=DuplicatingNodeElement, on='sites', variable=True, capacity=capacity)
        edges = GroupElement('links', element=Element, on='edges', variable=True, capacity=links)
        degree = MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1))
        return CompositeSystem(
            label,
            elements=[nodes, edges],
            interactions=[*changes, degree],
            space=space,
            time=DiscreteTime(),
        )

    def _build_system(self) -> CompositeSystem:
        """Return the steady-state duplication-deletion network of Jensen."""
        graph = nx.gnp_random_graph(self.n_nodes, self.density, seed=np.random.default_rng(self.random_state))
        death = TopologyInteraction('death', target='nodes', action='remove-node', one=True, probability=self.kill)
        birth = TopologyInteraction(
            'birth',
            target='nodes',
            action='duplicate-node',
            one=True,
            inherit=self.inherit,
            parent=self.parent,
            independent=True,
            toward=lambda node: self.new,
        )
        capacity = self.n_nodes * (2 * self.n_nodes - 1)
        return self._system(self.label, 2 * self.n_nodes, NetworkSpace(graph), capacity, [death, birth])
