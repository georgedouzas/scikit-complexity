"""Growing networks: preferential attachment, triad formation, and vertex copying."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import networkx as nx

from ....core.elements import GroupElement
from ....core.interactions import BaseInteraction, MapInteraction, TopologyInteraction
from ....core.laws.utils import neighbors, total, where
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ..._catalog import register_model
from ._copying_vertex import CopyingVertexElement
from ._link import LinkElement
from ._triad_vertex import TriadVertexElement
from ._vertex import DEGREE_VISUAL, VertexElement

SMALLEST_TOTAL = 1e-12


SMALLEST_DEGREE = 1e-9


JOINING_AGE = 0.5


def _degree(node: SimpleNamespace) -> object:
    """Return the degree of a candidate node, its weight under preferential attachment."""
    return neighbors(1)


def _get_bookkeeping() -> list[MapInteraction]:
    """Return the maps that age every node and read its degree from the edges at the start of a step."""
    return [
        MapInteraction('ageing', target='nodes', state='age', function=lambda s: 1, mode='increment'),
        MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1)),
    ]


def _get_triad_weight(node: SimpleNamespace) -> object:
    """Return the mixture of triad formation and preferential attachment for a candidate node."""
    closing = neighbors(node.other.near)
    attaching = neighbors(1)
    return node.triad * closing / (total(closing) + SMALLEST_TOTAL) + (1 - node.triad) * attaching / total(attaching)


def _nodes(element: type[VertexElement], capacity: int, **values: object) -> GroupElement:
    """Return the variable group on the nodes of a growing network, whose seed nodes start at age 1."""
    return GroupElement(
        'nodes',
        element=element,
        on='sites',
        variable=True,
        capacity=capacity,
        values={'age': 1.0, **values},
        visual=DEGREE_VISUAL,
    )


def _grown(
    nodes: GroupElement,
    growth: TopologyInteraction,
    space: NetworkSpace,
    capacity: int,
    interactions: Sequence[BaseInteraction] = (),
) -> System:
    """Return a growing network of a group on its nodes, a variable edge group, the arrival of nodes, and its maps."""
    links = GroupElement('links', element=LinkElement, on='edges', variable=True, capacity=capacity)
    return System(
        elements=[nodes, links],
        interactions=[growth, *_get_bookkeeping(), *interactions],
        space=space,
        time=DiscreteTime(),
    )


@register_model(
    field='mathematics/networks',
    tags=('network', 'growth', 'preferential attachment', 'scale-free'),
    references=('Barabási and Albert 1999', 'Argun et al. 2021', 'Gros 2015', 'Thurner et al. 2018'),
    defaults={'until': 300},
)
def barabasi_albert(n_nodes: int = 303, *, edges: int = 2, seed_nodes: int = 3) -> System:
    """Return Barabási-Albert growth, whose degrees follow a power law `P(k) ~ k^-3` by preferential attachment.

    The network starts complete on `seed_nodes` nodes, and every step one node joins with `edges` edges to nodes
    drawn with probability proportional to their degree, until it holds `n_nodes` nodes.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        seed_nodes:
            The number of nodes of the complete seed network.

    Returns:
        system:
            The model, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.mathematics.networks import barabasi_albert
        >>> trajectory = Simulation(barabasi_albert(13), random_state=0).check().prepare().run(until=10).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        23
    """
    space = NetworkSpace(nx.complete_graph(seed_nodes))
    nodes = _nodes(VertexElement, n_nodes)
    growth = TopologyInteraction('growth', action='add-node', edges=edges, toward=_degree)
    return _grown(nodes, growth, space, edges * n_nodes + seed_nodes**2)


@register_model(
    field='mathematics/networks',
    tags=('network', 'growth', 'preferential attachment', 'clustering', 'triad formation'),
    references=('Holme and Kim 2002', 'Argun et al. 2021', 'Gros 2015', 'Thurner et al. 2018'),
    defaults={'until': 300},
)
def holme_kim(n_nodes: int = 304, *, edges: int = 3, triad: float = 0.8, seed_nodes: int = 4) -> System:
    """Return Holme-Kim growth, scale-free like Barabási-Albert growth but with clustering that stays high.

    Every step a node joins with one edge by preferential attachment, and the next step it adds `edges - 1` edges,
    each to a neighbour of its first contact with probability `triad`, closing a triangle, and by preferential
    attachment otherwise.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        triad:
            The probability that an added edge closes a triangle.

        seed_nodes:
            The number of nodes of the complete seed network.

    Returns:
        system:
            The model, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.models.mathematics.networks import holme_kim
        >>> holme_kim(20).elements[0].element.__name__
        'TriadVertexElement'
    """
    space = NetworkSpace(nx.complete_graph(seed_nodes))
    nodes = _nodes(TriadVertexElement, n_nodes, triad=triad, extra=edges - 1)
    near = MapInteraction(
        'near',
        target='nodes',
        state='near',
        function=lambda s: neighbors(where(s.target.other.age < JOINING_AGE, 1, 0)),
    )
    closure = TopologyInteraction(
        'closure',
        target='nodes',
        action='add-edge',
        condition=lambda s: s.source.age < JOINING_AGE,
        toward=_get_triad_weight,
        edges=edges - 1,
    )
    growth = TopologyInteraction('growth', action='add-node', toward=_degree)
    return _grown(nodes, growth, space, edges * n_nodes + seed_nodes**2, [near, closure])


@register_model(
    field='mathematics/networks',
    tags=('network', 'growth', 'copying', 'scale-free'),
    references=('Kleinberg et al. 1999', 'Krapivsky and Redner 2005', 'Ramasco 2020'),
    defaults={'until': 300},
)
def vertex_copying(n_nodes: int = 304, *, edges: int = 3, randomness: float = 0.2, seed_nodes: int = 4) -> System:
    """Return vertex-copying growth, whose new nodes copy the edges of existing nodes and make a heavy-tailed degree.

    Every step a node joins with `edges` edges, each to a random node with probability `randomness` and otherwise to
    a random neighbour of a random prototype, which favours nodes of high degree without reading degrees.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        randomness:
            The probability that an edge joins a random node instead of copying.

        seed_nodes:
            The number of nodes of the complete seed network.

    Returns:
        system:
            The model, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.models.mathematics.networks import vertex_copying
        >>> vertex_copying(20).interactions[0].edges
        3
    """
    space = NetworkSpace(nx.complete_graph(seed_nodes))
    nodes = _nodes(CopyingVertexElement, n_nodes)
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        edges=edges,
        toward=lambda node: randomness + (1 - randomness) * neighbors(node.other.inverse),
    )
    inverse = MapInteraction(
        'inverse',
        target='nodes',
        state='inverse',
        function=lambda s: 1 / (neighbors(1) + SMALLEST_DEGREE),
    )
    return _grown(nodes, growth, space, edges * n_nodes + seed_nodes**2, [inverse])
