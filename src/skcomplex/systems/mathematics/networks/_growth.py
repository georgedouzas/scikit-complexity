"""Growing networks: preferential attachment, fitness, triad formation, copying, graphlets, and hyperbolic growth."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, exp, ln, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.interactions import BaseInteraction, MapInteraction, TopologyInteraction
from ....core.components.laws.utils import logistic, neighbors, total, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common._link import LinkElement
from ..common._vertex import DEGREE_VISUAL, VertexElement


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
        >>> from skcomplex.systems.mathematics.networks import CopyingVertexElement
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


class FitnessVertexElement(VertexElement):
    """Node of a growing network with an intrinsic fitness, which scales its chance to win new edges.

    A new edge joins the node with probability proportional to its fitness times its degree, as in the model of
    Bianconi and Barabási. The fitness is drawn when the node joins and stays fixed.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

        fitness:
            The fitness of the node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import FitnessVertexElement
        >>> FitnessVertexElement('nodes', fitness=0.5).fitness
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.fitness = fitness


class SimilarityVertexElement(VertexElement):
    """Node of a popularity-similarity network, with an angle on the circle next to its age and degree.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

        angle:
            The angle of the node on the circle, its similarity coordinate.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SimilarityVertexElement
        >>> SimilarityVertexElement('nodes', angle=1.5).angle
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        angle: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.angle = angle


class TriadVertexElement(VertexElement):
    """Node of a Holme-Kim network, which closes triangles with the neighbours of its first contact when it joins.

    One step after a node joins with one edge by preferential attachment, the triad formation of the system adds
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
        >>> from skcomplex.systems.mathematics.networks import TriadVertexElement
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
    tags=('network', 'growth', 'preferential attachment', 'scale-free'),
    references=(
        'Barabási and Albert 1999',
        'Bianconi and Barabási 2001',
        'Argun et al. 2021',
        'Gros 2015',
        'Thurner et al. 2018',
        'Raman 2021',
    ),
    defaults={'until': 300},
)
def barabasi_albert(
    n_nodes: int = 303,
    *,
    edges: int = 2,
    seed_nodes: int = 3,
    fitness: bool = False,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Barabási-Albert growth, whose degrees follow a power law `P(k) ~ k^-3` by preferential attachment.

    The network starts complete on `seed_nodes` nodes, and every step one node joins with `edges` edges to nodes
    drawn with probability proportional to their degree, until it holds `n_nodes` nodes.

    With `fitness`, every node draws a fitness `eta` uniformly in `[0, 1]` and new edges join nodes with probability
    proportional to `eta k`, the model of Bianconi and Barabási (2001) that Raman (2021, section 3.2.4) describes.
    The degree of a node then grows as `t^beta(eta)` with an exponent that rises with its fitness, so a fit late node
    overtakes older ones.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        seed_nodes:
            The number of nodes of the complete seed network.

        fitness:
            Whether the nodes draw a fitness that scales their attachment, the Bianconi-Barabási model.

        random_state:
            The seed or generator of the fitness of the seed nodes.

    Returns:
        system:
            The system, with the groups `nodes` and `links` and the arrival `growth`, and with `fitness` the state
            `nodes.fitness`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import barabasi_albert
        >>> trajectory = Simulation(barabasi_albert(13), random_state=0).prepare().run(until=10).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        23
    """
    space = NetworkSpace(nx.complete_graph(seed_nodes))
    capacity = edges * n_nodes + seed_nodes**2
    if not fitness:
        nodes = _nodes(VertexElement, n_nodes)
        growth = TopologyInteraction('growth', action='add-node', edges=edges, toward=_degree)
        return _grown(nodes, growth, space, capacity)
    seed_fitness = np.random.default_rng(random_state).uniform(0.0, 1.0, seed_nodes)
    nodes = _nodes(FitnessVertexElement, n_nodes, fitness=seed_fitness.tolist())
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        edges=edges,
        values={'fitness': lambda rng: rng.uniform(0.0, 1.0)},
        toward=lambda node: node.fitness * neighbors(1),
    )
    return _grown(nodes, growth, space, capacity)


@register_system(
    field='mathematics/networks',
    tags=('network', 'growth', 'preferential attachment', 'clustering', 'triad formation'),
    references=('Holme and Kim 2002', 'Downey 2018'),
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


@register_system(
    field='mathematics/networks',
    tags=('network', 'growth', 'graphlet', 'preferential attachment', 'tree'),
    references=('Filkov et al. 2009', 'Raman 2021'),
    defaults={'until': 300},
)
def graphlet_arrival(n_nodes: int = 602, *, alpha: float = 0.3) -> System:
    """Return growth by the arrival of three-node graphlets, a tree whose degree exponent rises with `alpha`.

    The network starts from one edge, and every step a graphlet of a central node and two peripheral nodes joined to it
    arrives and merges one of its nodes with an existing node drawn with probability proportional to its degree: a
    peripheral node with probability `alpha`, so the existing node gains the central node, which carries the other
    peripheral node, and the central node otherwise, so the existing node gains both peripheral nodes (Filkov et al.
    2009, as Raman 2021, section 3.2.4.2, describes). Every step adds two nodes and two edges, and the shares of nodes
    of degree 1 and 2 are `2 (2 - alpha) / 5` and `alpha (7 - alpha) / 15`.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        alpha:
            The probability that the graphlet attaches by a peripheral node.

    Returns:
        system:
            The system, with the groups `nodes` and `links` and the arrivals `growth` and `graphlet`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import graphlet_arrival
        >>> trajectory = Simulation(graphlet_arrival(22), random_state=0).prepare().run(until=10).trajectory_
        >>> int(trajectory.network(-1).nodes.size), int(trajectory.network(-1).sources.size)
        (22, 21)
    """
    nodes = _nodes(VertexElement, n_nodes)
    growth = TopologyInteraction('growth', action='add-node', toward=_degree)
    graphlet = TopologyInteraction(
        'graphlet',
        action='add-node',
        values={'age': 0.0},
        toward=lambda node: alpha * where(node.age < JOINING_AGE, 1, 0)
        + (1 - alpha) * where(node.age > JOINING_AGE, 1, 0) * neighbors(where(node.other.age < JOINING_AGE, 1, 0)),
    )
    return _grown(nodes, growth, NetworkSpace(nx.path_graph(2)), n_nodes, [graphlet])
