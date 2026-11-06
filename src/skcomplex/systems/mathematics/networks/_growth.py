"""Growing networks: preferential attachment, triad formation, vertex copying, and popularity-similarity growth."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import exp, ln, pi, sin

from ....core.components.elements import GroupElement
from ....core.components.interactions import BaseInteraction, MapInteraction, TopologyInteraction
from ....core.components.laws.utils import logistic, neighbors, total, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ..._catalog import register_system
from ._copying_vertex import CopyingVertexElement
from ._link import LinkElement
from ._similarity_vertex import SimilarityVertexElement
from ._triad_vertex import TriadVertexElement
from ._vertex import DEGREE_VISUAL, VertexElement

SMALLEST_TOTAL = 1e-12


SMALLEST_DEGREE = 1e-9


JOINING_AGE = 0.5


SMALLEST_ANGLE = 1e-12


MIN_EXPONENT = 2.0


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


@register_system(
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
            The system, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import barabasi_albert
        >>> trajectory = Simulation(barabasi_albert(13), random_state=0).prepare().run(until=10).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        23
    """
    space = NetworkSpace(nx.complete_graph(seed_nodes))
    nodes = _nodes(VertexElement, n_nodes)
    growth = TopologyInteraction('growth', action='add-node', edges=edges, toward=_degree)
    return _grown(nodes, growth, space, edges * n_nodes + seed_nodes**2)


@register_system(
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
            The system, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import holme_kim
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


@register_system(
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
            The system, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import vertex_copying
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


def _get_similarity_weight(
    node: SimpleNamespace,
    arrivals: int,
    *,
    edges: int,
    fading: float,
    temperature: float,
) -> object:
    """Return the Fermi-Dirac probability of the hyperbolic distance between a candidate node and the new node."""
    now = node.t + arrivals
    radius = 2 * ln(now)
    candidate = fading * 2 * ln(now - node.age) + (1 - fading) * radius
    angle = pi - abs(pi - abs(node.angle - node.source.angle))
    distance = candidate + radius + 2 * ln(angle / 2 + SMALLEST_ANGLE)
    spread = 2 * temperature * (1 - exp(-(1 - fading) * radius / 2)) / (sin(temperature * pi) * edges * (1 - fading))
    return 1 - logistic(distance, radius - 2 * ln(spread), 2 * temperature)


@register_system(
    field='mathematics/networks',
    tags=('network', 'growth', 'hyperbolic', 'popularity-similarity', 'scale-free', 'clustering'),
    references=(
        'Papadopoulos et al. 2012, arXiv 1106.0286',
        'Gabrielli, Garlaschelli, Patil, and Serrano 2024, arXiv 2412.12988',
    ),
    defaults={'until': 297},
)
def popularity_similarity_network(
    n_nodes: int = 300,
    *,
    edges: int = 2,
    exponent: float = 2.5,
    temperature: float = 0.2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return popularity-similarity growth, scale-free and clustered, whose nodes attach by hyperbolic distance.

    Every step a node joins at a uniform angle on the circle and the radius `2 ln t`, every other node `s` moves to the
    radius `b 2 ln s + (1 - b) 2 ln t` with `b = 1 / (exponent - 1)`, and the new node joins `edges` nodes drawn in
    proportion to the Fermi-Dirac probability of their hyperbolic distance `x = r_s + r_t + 2 ln(theta / 2)`, at the
    temperature `temperature`. The degrees follow a power law of the given exponent, and the clustering falls as the
    temperature rises (Papadopoulos et al. 2012).

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node, and one less than the nodes of the complete seed network.

        exponent:
            The exponent of the degree distribution, above 2.

        temperature:
            The temperature of the network, between 0 and 1.

        random_state:
            The seed or generator of the angles of the seed network.

    Returns:
        system:
            The system, with the groups `nodes` and `links`, the arrival `growth`, and the states `nodes.angle`,
            `nodes.age`, and `nodes.degree`.

    Raises:
        ValidationError:
            The exponent is not above 2, or the temperature is not between 0 and 1.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import popularity_similarity_network
        >>> system = popularity_similarity_network(13)
        >>> trajectory = Simulation(system, random_state=0).prepare().run(until=10).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        23
    """
    if exponent <= MIN_EXPONENT:
        error_msg = f'The degree exponent of the growth is {exponent!r}. Give an exponent above {MIN_EXPONENT}.'
        raise ValidationError(error_msg)
    if not 0 < temperature < 1:
        error_msg = f'The temperature of the growth is {temperature!r}. Give a temperature between 0 and 1.'
        raise ValidationError(error_msg)
    seed_nodes = edges + 1
    fading = 1 / (exponent - 1)
    angles = np.random.default_rng(random_state).uniform(0.0, 2 * np.pi, seed_nodes)
    ages = np.arange(seed_nodes - 1, -1, -1, dtype=float)
    nodes = _nodes(SimilarityVertexElement, n_nodes, age=ages.tolist(), angle=angles.tolist())
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        edges=edges,
        values={'angle': lambda rng: rng.uniform(0.0, 2 * np.pi)},
        toward=lambda node: _get_similarity_weight(
            node,
            seed_nodes + 1,
            edges=edges,
            fading=fading,
            temperature=temperature,
        ),
    )
    return _grown(nodes, growth, NetworkSpace(nx.complete_graph(seed_nodes)), edges * n_nodes + seed_nodes**2)
