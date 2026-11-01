"""Growing networks: preferential attachment, triad formation, and vertex copying."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import BaseInteraction, MapInteraction, TopologyInteraction
from ....core.laws.utils import neighbors, total, where
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})
SMALLEST_TOTAL = 1e-12
SMALLEST_DEGREE = 1e-9
JOINING_AGE = 0.5


def _degree(node: SimpleNamespace) -> object:
    """Return the degree of a candidate node, its weight under preferential attachment."""
    return neighbors(1)


class LinkElement(Element):
    """Edge of a growing network, which holds no state of its own.

    Args:
        label:
            The label of the edge group.

    Examples:
        >>> from skcomplex.models.mathematics.networks import LinkElement
        >>> LinkElement('links').label
        'links'
    """


class VertexElement(Element):
    """Node of a growing network, with its age in steps and its degree.

    New nodes start with age 0, and the maps of the model age the node and read its degree from the edges at the start
    of every step.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

    Examples:
        >>> from skcomplex.models.mathematics.networks import VertexElement
        >>> VertexElement('nodes').age
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = DEGREE_VISUAL,
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
        self.degree = degree


class TriadVertexElement(VertexElement):
    """Node of a Holme-Kim network, which closes triangles with the neighbours of its first contact when it joins.

    One step after a node joins with one edge by preferential attachment, the triad formation of the model adds
    `extra` edges to it. Each new end is a neighbour of its first contact with probability `triad`, and a node drawn by
    preferential attachment otherwise. The state `near` marks the nodes next to the newest node.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

        near:
            The number of neighbours of the node that joined in the previous step.

        triad:
            The probability that a new edge closes a triangle.

        extra:
            The number of edges a node adds after its first one.

    Examples:
        >>> from skcomplex.models.mathematics.networks import TriadVertexElement
        >>> TriadVertexElement('nodes', triad=0.5).triad
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        near: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        triad: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        extra: Annotated[int, ParameterSpec(domain=NN)] = 1,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.near = near
        self.triad = triad
        self.extra = extra


class CopyingVertexElement(VertexElement):
    """Node of a vertex-copying network, which holds the inverse of its degree for the copying of its edges.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

        inverse:
            The inverse of the degree, which only the neighbours of the node read.

    Examples:
        >>> from skcomplex.models.mathematics.networks import CopyingVertexElement
        >>> CopyingVertexElement('nodes').inverse
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        inverse: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.inverse = inverse


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
