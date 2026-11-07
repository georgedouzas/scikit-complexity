"""Hypergraphs: the size-biased random walk across hyperedges, and the simplicial complex that grows by triangles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import HyperedgeInteraction, MapInteraction, TopologyInteraction
from ....core.components.laws import BaseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import neighbors, total
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common._vertex import VertexElement


class SimplexElement(Element):
    """Triangle of a growing simplicial complex, with its age in steps.

    Args:
        label:
            The label of the triangle or the group on the triangles.

        visual:
            How the triangle is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the triangle joined.

    Attributes:
        default_visual_:
            The visual of the triangle when it sets none.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SimplexElement
        >>> SimplexElement('triangles').age
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age


class WalkerDensityElement(Element):
    """Probability that a random walker is at a node of a hypergraph, with the weight `D` of the node.

    Args:
        label:
            The label of the node or the group on the nodes.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        p:
            The probability of the walker at the node.

        D:
            The total weight of the moves out of the node, the sum of `(|e| - 1)^2` over its hyperedges `e`.

    Attributes:
        default_visual_:
            The visual of the node when it sets none.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import WalkerDensityElement
        >>> WalkerDensityElement('node', D=4.0).D
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p
        self.D = D


class SimplexAgeingInteraction(HyperedgeInteraction):
    """Ageing of every triangle of a simplicial complex by one per step.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes.

        triangles:
            The path of the group on the triangles.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SimplexAgeingInteraction
        >>> SimplexAgeingInteraction('ageing', nodes='nodes', triangles='triangles').size
        3
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        triangles: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label, size=3)
        self.nodes = nodes
        self.triangles = triangles

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the ageing of the triangle."""
        return [UpdateLaw(s.hyperedge.age, s.hyperedge.age + 1)]


class SizeBiasedMoveInteraction(HyperedgeInteraction):
    """Moves of a random walker across a hyperedge, to every other member with weight `|e| - 1`.

    A walker at node `i` moves to node `j` at rate `T_ij = k_ij / D_i`, where `k_ij = sum_e (|e| - 1)` over the
    hyperedges `e` that hold both, and `D_i = sum_j k_ij`. On a hyperedge, the focal member gains `(|e| - 1) p_j / D_j`
    from every other member `j` and loses `(|e| - 1)^2 p_i / D_i`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of walker densities on the nodes.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SizeBiasedMoveInteraction
        >>> SizeBiasedMoveInteraction('moves', target='nodes').target
        'nodes'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flow into and out of the focal member across one hyperedge."""
        weight = len(s.others)
        member = s.member
        inflow = total(s.members.p / s.members.D, over=s.members) - member.p / member.D
        return [RateLaw(member.p, weight * inflow - weight**2 * member.p / member.D)]


class HypergraphRandomWalkSystem(BaseSystem):
    """The size-biased random walk on a hypergraph, `p_j' = sum_i p_i T_ij - p_j`, from one node.

    A walker at node `i` moves to node `j` at rate `T_ij = k_ij / sum_l k_il`, with `k_ij = sum_e (|e| - 1) e_i e_j`
    over the hyperedges `e` of the incidence matrix, so large hyperedges carry more of the walk, as in Carletti et al.
    The probability of the walker spreads from the source, keeps its sum of 1, and settles on the stationary
    distribution `p_i = D_i / sum_l D_l`, with `D_i = sum_e (|e| - 1)^2 e_i` the weight of the hyperedges of `i`. The
    system has the probabilities `nodes.p` and the weights `nodes.D`.

    Args:
        hypergraph:
            The XGI hypergraph, or `None` for a random hypergraph of 30 nodes with hyperedges of sizes 2 to 5.

        source:
            The node the walker starts at.

        seed:
            The seed of the random hypergraph.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import xgi
        >>> from skcomplex.systems.mathematics.networks import HypergraphRandomWalkSystem
        >>> system = HypergraphRandomWalkSystem(xgi.Hypergraph([[0, 1, 2], [2, 3]])).prepare().system_
        >>> system.elements[0].values['D'].tolist()
        [4.0, 4.0, 5.0, 1.0]
    """

    field = 'mathematics/networks'
    topics = ('hypergraph', 'random walk', 'master equation', 'ode')
    references = ('Carletti et al. 2020', 'Majhi et al. 2022')
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        hypergraph: xgi.Hypergraph | None = None,
        *,
        source: int = 0,
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.hypergraph = hypergraph
        self.source = source
        self.seed = seed

    def _build_system(self) -> System:
        """Return the size-biased random walk on a hypergraph, `p_j' = sum_i p_i T_ij - p_j`, from one node."""
        hypergraph = self.hypergraph
        if hypergraph is None:
            hypergraph = xgi.fast_random_hypergraph(30, [0.1, 0.01, 0.001, 0.0005], seed=self.seed)
        space = NetworkSpace(nx.empty_graph(sorted(hypergraph.nodes)), hyperedges=hypergraph)
        prepared = space.prepare()
        weights = np.zeros(prepared.size_)
        for hyperedge in prepared.hyperedges_:
            weights[list(hyperedge)] += (len(hyperedge) - 1) ** 2
        probabilities = np.zeros(prepared.size_)
        probabilities[self.source] = 1.0
        nodes = LatticeElement(
            'nodes',
            element=WalkerDensityElement,
            space=space,
            values={'p': probabilities, 'D': np.maximum(weights, 1.0)},
            visual=VisualSpec('sites', {'states': ['p'], 'low': 0.0, 'high': 1.0}),
        )
        return System(self.label, elements=[nodes], interactions=[SizeBiasedMoveInteraction('moves', target='nodes')])


class GrowingSimplicialComplexSystem(BaseSystem):
    """A simplicial complex that grows by gluing one triangle per step along an edge drawn uniformly.

    The complex starts as one triangle, and every step one node joins and forms a triangle with the ends of an edge
    drawn uniformly, adding two edges, as the network geometry with flavor `s = 0` of Bianconi and Rahmede in two
    dimensions. A node gains triangles in proportion to its degree, so the degrees follow the power law `P(k) ~ k^-3` of
    preferential attachment, and every triangle ages by one per step. The system has the groups `nodes`, `links`, and
    `triangles`, and the arrival `growth`.

    Args:
        n_nodes:
            The number of nodes the complex holds room for.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import GrowingSimplicialComplexSystem
        >>> simulate(GrowingSimplicialComplexSystem(13), until=10, random_state=0).network(-1).hyperedges.shape
        (11, 3)
    """

    field = 'mathematics/networks'
    topics = ('simplicial complex', 'hypergraph', 'growth', 'network geometry', 'scale-free')
    references = ('Bianconi and Rahmede 2016', 'Majhi et al. 2022')
    until = 300

    def __init__(
        self,
        n_nodes: int = 303,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes

    @staticmethod
    def _get_bookkeeping() -> list[MapInteraction]:
        """Return the maps that age every node and read its degree from the edges at the start of a step."""
        return [
            MapInteraction('ageing', target='nodes', state='age', function=lambda s: 1, mode='increment'),
            MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1)),
        ]

    def _build_system(self) -> System:
        """Return a simplicial complex that grows by gluing one triangle per step along an edge drawn uniformly."""
        nodes = GroupElement(
            'nodes',
            element=VertexElement,
            on='sites',
            variable=True,
            capacity=self.n_nodes,
            values={'age': 1.0},
        )
        links = GroupElement('links', element=Element, on='edges', variable=True, capacity=2 * self.n_nodes)
        triangles = GroupElement(
            'triangles',
            element=SimplexElement,
            on='hyperedges',
            variable=True,
            capacity=self.n_nodes,
        )
        growth = TopologyInteraction('growth', action='add-node', hyperedge=3)
        ageing = SimplexAgeingInteraction('triangle_ageing', nodes='nodes', triangles='triangles')
        return System(
            self.label,
            elements=[nodes, links, triangles],
            interactions=[growth, ageing, *self._get_bookkeeping()],
            space=NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]]),
            time=DiscreteTime(),
        )
