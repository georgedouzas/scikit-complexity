"""Hypergraphs: the size-biased random walk across hyperedges, and the simplicial complex that grows by triangles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import networkx as nx
import numpy as np
import xgi

from ....core.components.elements import GroupElement, LatticeElement
from ....core.components.interactions import TopologyInteraction
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.specs import VisualSpec
from ..._catalog import register_system
from ._growth import _get_bookkeeping
from ._link import LinkElement
from ._simplex import SimplexElement
from ._simplex_ageing import SimplexAgeingInteraction
from ._size_biased_move import SizeBiasedMoveInteraction
from ._vertex import VertexElement
from ._walker_density import WalkerDensityElement


@register_system(
    field='mathematics/networks',
    tags=('hypergraph', 'random walk', 'master equation', 'ode'),
    references=('Carletti et al. 2020', 'Majhi et al. 2022'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def hypergraph_random_walk(
    hypergraph: xgi.Hypergraph | None = None,
    *,
    source: int = 0,
    seed: int = 0,
) -> System:
    """Return the size-biased random walk on a hypergraph, `p_j' = sum_i p_i T_ij - p_j`, from one node.

    A walker at node `i` moves to node `j` at rate `T_ij = k_ij / sum_l k_il`, with `k_ij = sum_e (|e| - 1) e_i e_j`
    over the hyperedges `e` of the incidence matrix, so large hyperedges carry more of the walk, as in Carletti et al.
    The probability of the walker spreads from the source, keeps its sum of 1, and settles on the stationary
    distribution `p_i = D_i / sum_l D_l`, with `D_i = sum_e (|e| - 1)^2 e_i` the weight of the hyperedges of `i`.

    Args:
        hypergraph:
            The XGI hypergraph, or `None` for a random hypergraph of 30 nodes with hyperedges of sizes 2 to 5.

        source:
            The node the walker starts at.

        seed:
            The seed of the random hypergraph.

    Returns:
        system:
            The system, with the probabilities `nodes.p` and the weights `nodes.D`.

    Examples:
        >>> import xgi
        >>> from skcomplex.systems.mathematics.networks import hypergraph_random_walk
        >>> system = hypergraph_random_walk(xgi.Hypergraph([[0, 1, 2], [2, 3]]))
        >>> system.elements[0].values['D'].tolist()
        [4.0, 4.0, 5.0, 1.0]
    """
    if hypergraph is None:
        hypergraph = xgi.fast_random_hypergraph(30, [0.1, 0.01, 0.001, 0.0005], seed=seed)
    space = NetworkSpace(nx.empty_graph(sorted(hypergraph.nodes)), hyperedges=hypergraph)
    prepared = space.prepare()
    weights = np.zeros(prepared.size_)
    for hyperedge in prepared.hyperedges_:
        weights[list(hyperedge)] += (len(hyperedge) - 1) ** 2
    probabilities = np.zeros(prepared.size_)
    probabilities[source] = 1.0
    nodes = LatticeElement(
        'nodes',
        element=WalkerDensityElement,
        space=space,
        values={'p': probabilities, 'D': np.maximum(weights, 1.0)},
        visual=VisualSpec('sites', {'states': ['p'], 'low': 0.0, 'high': 1.0}),
    )
    return System(elements=[nodes], interactions=[SizeBiasedMoveInteraction('moves', target='nodes')])


@register_system(
    field='mathematics/networks',
    tags=('simplicial complex', 'hypergraph', 'growth', 'network geometry', 'scale-free'),
    references=('Bianconi and Rahmede 2016', 'Majhi et al. 2022'),
    defaults={'until': 300},
)
def growing_simplicial_complex(n_nodes: int = 303) -> System:
    """Return a simplicial complex that grows by gluing one triangle per step along an edge drawn uniformly.

    The complex starts as one triangle, and every step one node joins and forms a triangle with the ends of an edge
    drawn uniformly, adding two edges, as the network geometry with flavor `s = 0` of Bianconi and Rahmede in two
    dimensions. A node gains triangles in proportion to its degree, so the degrees follow the power law
    `P(k) ~ k^-3` of preferential attachment, and every triangle ages by one per step.

    Args:
        n_nodes:
            The number of nodes the complex holds room for.

    Returns:
        system:
            The system, with the groups `nodes`, `links`, and `triangles`, and the arrival `growth`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import growing_simplicial_complex
        >>> simulation = Simulation(growing_simplicial_complex(13), random_state=0).prepare()
        >>> simulation.run(until=10).trajectory_.network(-1).hyperedges.shape
        (11, 3)
    """
    nodes = GroupElement(
        'nodes',
        element=VertexElement,
        on='sites',
        variable=True,
        capacity=n_nodes,
        values={'age': 1.0},
    )
    links = GroupElement('links', element=LinkElement, on='edges', variable=True, capacity=2 * n_nodes)
    triangles = GroupElement('triangles', element=SimplexElement, on='hyperedges', variable=True, capacity=n_nodes)
    growth = TopologyInteraction('growth', action='add-node', hyperedge=3)
    ageing = SimplexAgeingInteraction('triangle_ageing', nodes='nodes', triangles='triangles')
    return System(
        elements=[nodes, links, triangles],
        interactions=[growth, ageing, *_get_bookkeeping()],
        space=NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]]),
        time=DiscreteTime(),
    )
