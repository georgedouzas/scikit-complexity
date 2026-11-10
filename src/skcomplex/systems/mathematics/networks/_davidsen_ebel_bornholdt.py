"""Davidsen-Ebel-Bornholdt network.

Triadic closure and node turnover at a fixed size give high clustering.
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


class ClosingNodeElement(Element):
    """Node of the Davidsen-Ebel-Bornholdt network, with its degree.

    A map of the system reads the degree from the edges at the start of every step.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        degree:
            The number of edges of the node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import ClosingNodeElement
        >>> ClosingNodeElement('nodes').degree
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


class DavidsenEbelBornholdtSystem(BaseSystem):
    """The Davidsen-Ebel-Bornholdt network.

    Its triadic closure and node turnover give high clustering.

    The network starts as an Erdős-Rényi graph of `n_nodes` nodes and `links` links, `n_nodes` by default. Every step a
    node drawn uniformly joins two of its neighbours that are not yet joined, or, without such a pair, joins a node
    drawn uniformly. Then, with probability `turnover`, a node drawn uniformly leaves with its links and a new node
    takes its place with one link to a node drawn uniformly (Thurner et al. 2018, section 4.5.4.6). A closure always
    adds one link, and a turnover removes a node of mean degree `k` and adds one link, so the steady state has the mean
    degree `k = 1 + 1 / turnover`. Closing triangles keeps the clustering high, and since a node lies near many others
    in proportion to its degree, closure attaches preferentially, so hubs emerge as the turnover falls. The system has
    the groups `nodes` and `links`, the closure `closure`, and the turnover `turnover`.

    Args:
        n_nodes:
            The number of nodes.

        turnover:
            The probability that a node is replaced at a step.

        links:
            The number of links of the initial Erdős-Rényi graph, or `None` for `n_nodes`.

        random_state:
            The seed or generator of the initial graph.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import DavidsenEbelBornholdtSystem
        >>> trajectory = simulate(DavidsenEbelBornholdtSystem(50), until=20, random_state=0)
        >>> len(trajectory.get_network(-1).nodes)
        50
    """

    field = 'mathematics/networks'
    topics = ('network', 'triadic closure', 'clustering', 'node turnover', 'non-growing network')
    references = ('Davidsen, Ebel, and Bornholdt 2002', 'Thurner et al. 2018')
    until = 10000

    def __init__(
        self,
        n_nodes: int = 200,
        *,
        turnover: float = 0.1,
        links: int | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.turnover = turnover
        self.links = links
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
        nodes = GroupElement('nodes', element=ClosingNodeElement, on='sites', variable=True, capacity=capacity)
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
        """Return the Davidsen-Ebel-Bornholdt network.

        Its triadic closure and node turnover give high clustering.
        """
        rng = np.random.default_rng(self.random_state)
        graph = nx.gnm_random_graph(self.n_nodes, self.n_nodes if self.links is None else self.links, seed=rng)
        closure = TopologyInteraction('closure', target='nodes', action='close-triangle', one=True)
        replacement = TopologyInteraction(
            'turnover',
            target='nodes',
            action='replace-node',
            one=True,
            probability=self.turnover,
        )
        capacity = graph.number_of_edges() + int(2 * self.n_nodes * (1 + 1 / self.turnover))
        return self._system(self.label, self.n_nodes, NetworkSpace(graph), capacity, [closure, replacement])
