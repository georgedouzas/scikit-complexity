"""Hypergraphs.

The module covers the size-biased random walk across hyperedges, and the simplicial complex that grows by triangles.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw, TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import neighbors, total
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.networks import VertexElement


class SimplexElement(Element):
    """Triangle of a growing simplicial complex, with its age in steps.

    Args:
        label:
            The label of the triangle or the group on the triangles.

        age:
            The number of steps since the triangle joined.

        space:
            The network whose hyperedges hold one triangle each, or `None` for one triangle.

    Attributes:
        default_visual_:
            How the triangle is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SimplexElement
        >>> SimplexElement('triangles').age
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        age: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space, on='hyperedges' if space is not None else None)
        self.age = age


class WalkerDensityElement(Element):
    """Probability that a random walker is at a node of a hypergraph.

    The node has the weight `D`.

    Args:
        label:
            The label of the node or the group on the nodes.

        p:
            The probability of the walker at the node.

        D:
            The total weight of the moves out of the node, the sum of `(|e| - 1)^2` over its hyperedges `e`.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Attributes:
        default_visual_:
            How the node is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import WalkerDensityElement
        >>> WalkerDensityElement('node', D=4.0).D
        4.0
    """

    visual = VisualSpec('sites', {'states': ['p'], 'low': 0.0, 'high': 1.0})

    def __init__(
        self,
        label: str,
        *,
        p: Annotated[object, RealSet.closed(0, 1)] = 0.0,
        D: Annotated[object, RealSet.open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.p = p
        self.D = D


class SimplexAgeingInteraction(Interaction):
    """Ageing of every triangle of a simplicial complex by one per step.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group on the triangles.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SimplexAgeingInteraction
        >>> SimplexAgeingInteraction('ageing').target
        'triangles'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'triangles') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the ageing of every triangle.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        triangles = context.target
        self.laws_: list[BaseComponent] = [UpdateLaw(triangles.age, triangles.age + 1)]


class SizeBiasedMoveInteraction(Interaction):
    """Moves of a random walker across a hyperedge.

    The walker moves to every other member with weight `|e| - 1`.

    A walker at node `i` moves to node `j` at rate `T_ij = k_ij / D_i`, where `k_ij = sum_e (|e| - 1)` over the
    hyperedges `e` that hold both, and `D_i = sum_j k_ij`. On a hyperedge, the focal member gains `(|e| - 1) p_j / D_j`
    from every other member `j` and loses `(|e| - 1)^2 p_i / D_i`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        among:
            The path of the group of walker densities on the nodes.

        along:
            `'hyperedges'`, the hyperedges the walker moves across.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SizeBiasedMoveInteraction
        >>> SizeBiasedMoveInteraction('moves', among='nodes').among
        'nodes'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, among: Role = 'nodes', along: str = 'hyperedges') -> None:
        super().__init__(label, among=among, along=along)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the flow into and out of the focal member across one hyperedge.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        s = context
        weight = len(s.others)
        member = s.member
        inflow = total(s.members.p / s.members.D, over=s.members) - member.p / member.D
        self.laws_: list[BaseComponent] = [
            RateLaw(member.p.diff(s.t), weight * inflow - weight**2 * member.p / member.D),
        ]


class HypergraphRandomWalkSystem(BaseSystem):
    """The size-biased random walk on a hypergraph, `p_j' = sum_i p_i T_ij - p_j`.

    The walk starts from one node.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import xgi
        >>> from skcomplex.systems.mathematics.networks import HypergraphRandomWalkSystem
        >>> system = HypergraphRandomWalkSystem(xgi.Hypergraph([[0, 1, 2], [2, 3]])).prepare().system_
        >>> system.elements[0].D.tolist()
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.hypergraph = hypergraph
        self.source = source
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the size-biased random walk `p_j' = sum_i p_i T_ij - p_j` from one node."""
        super()._prepare(context)
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
        nodes = WalkerDensityElement('nodes', p=probabilities, D=np.maximum(weights, 1.0), space=space)
        self.system_ = System(self.label, elements=[nodes], interactions=[SizeBiasedMoveInteraction('moves')])


class GrowingSimplicialComplexSystem(BaseSystem):
    """A simplicial complex that grows by gluing one triangle per step along an edge.

    The edge is drawn uniformly.

    The complex starts as one triangle, and every step one node joins and forms a triangle with the ends of an edge
    drawn uniformly, adding two edges, as the network geometry with flavor `s = 0` of Bianconi and Rahmede in two
    dimensions. A node gains triangles in proportion to its degree, so the degrees follow the power law `P(k) ~ k^-3` of
    preferential attachment, and every triangle ages by one per step. The system has the groups `nodes`, `links`, and
    `triangles`, and the arrival `growth`.

    Args:
        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import GrowingSimplicialComplexSystem
        >>> trajectory = GrowingSimplicialComplexSystem().prepare().simulate(until=10, random_state=0)
        >>> trajectory.get_network(-1).hyperedges.shape
        (11, 3)
    """

    field = 'mathematics/networks'
    topics = ('simplicial complex', 'hypergraph', 'growth', 'network geometry', 'scale-free')
    references = ('Bianconi and Rahmede 2016', 'Majhi et al. 2022')
    until = 300

    def __init__(
        self,
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)

    @staticmethod
    def _get_bookkeeping() -> list[Interaction]:
        """Return the updates that age every node and read its degree from the edges.

        The updates run at the start of a step.
        """
        return [
            Interaction(
                'ageing',
                roles={'target': 'nodes'},
                laws=lambda s: [UpdateLaw(s.target.age, s.target.age + 1, increment=True)],
            ),
            Interaction(
                'degree',
                roles={'target': 'nodes'},
                laws=lambda s: [UpdateLaw(s.target.degree, neighbors(1))],
            ),
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to a simplicial complex that grows by gluing one triangle per step along an edge."""
        super()._prepare(context)
        space = NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]])
        nodes = VertexElement('nodes', age=1.0, space=space)
        links = Element('links', space=space, on='edges')
        triangles = SimplexElement('triangles', space=space)
        request = TopologyRequestSpec('add-node', hyperedge=3, values=(('age', 0.0),))
        growth = Interaction('growth', laws=[UpdateLaw(request, 1)])
        ageing = SimplexAgeingInteraction('triangle_ageing')
        self.system_ = System(
            self.label,
            elements=[nodes, links, triangles],
            interactions=[growth, ageing, *self._get_bookkeeping()],
            space=space,
        )
