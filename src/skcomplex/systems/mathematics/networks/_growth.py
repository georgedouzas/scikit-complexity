"""Growing networks.

The module covers preferential attachment, fitness, triad formation, copying, graphlets, and hyperbolic growth.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, exp, ln, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import BaseInteraction, MapInteraction, TopologyInteraction
from ....core.components.laws.utils import logistic, neighbors, total, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.networks import VertexElement
from ..common.networks._vertex import DEGREE_VISUAL

SMALLEST_TOTAL = 1e-12
SMALLEST_DEGREE = 1e-9
JOINING_AGE = 0.5
SMALLEST_ANGLE = 1e-12
MIN_EXPONENT = 2.0


class CopyingVertexElement(VertexElement):
    """Node of a vertex-copying network.

    The node holds the inverse of its degree for the copying of its edges.

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
    """Node of a growing network with an intrinsic fitness.

    The fitness scales its chance to win new edges.

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
    """Node of a popularity-similarity network.

    The node has an angle on the circle next to its age and degree.

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
    """Node of a Holme-Kim network.

    When it joins, the node closes triangles with the neighbours of its first contact.

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


class BarabasiAlbertSystem(BaseSystem):
    """Barabási-Albert growth.

    By preferential attachment the degrees follow a power law `P(k) ~ k^-3`.

    The network starts complete on `seed_nodes` nodes, and every step one node joins with `edges` edges to nodes
    drawn with probability proportional to their degree, until it holds `n_nodes` nodes.

    With `fitness`, every node draws a fitness `eta` uniformly in `[0, 1]` and new edges join nodes with probability
    proportional to `eta k`, the model of Bianconi and Barabási (2001) that Raman (2021, section 3.2.4) describes. The
    degree of a node then grows as `t^beta(eta)` with an exponent that rises with its fitness, so a fit late node
    overtakes older ones. The system has the groups `nodes` and `links` and the arrival `growth`, and with `fitness` the
    state `nodes.fitness`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import BarabasiAlbertSystem
        >>> trajectory = simulate(BarabasiAlbertSystem(13), until=10, random_state=0)
        >>> int(trajectory.network(-1).sources.size)
        23
    """

    field = 'mathematics/networks'
    topics = ('network', 'growth', 'preferential attachment', 'scale-free')
    references = (
        'Barabási and Albert 1999',
        'Bianconi and Barabási 2001',
        'Argun et al. 2021',
        'Gros 2015',
        'Thurner et al. 2018',
        'Raman 2021',
    )
    until = 300

    @staticmethod
    def _get_degree(node: SimpleNamespace) -> object:
        """Return the degree of a candidate node, its weight under preferential attachment."""
        return neighbors(1)

    @staticmethod
    def _get_bookkeeping() -> list[MapInteraction]:
        """Return the maps that age every node and read its degree from the edges.

        The maps run at the start of a step.
        """
        return [
            MapInteraction('ageing', target='nodes', state='age', function=lambda s: 1, mode='increment'),
            MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1)),
        ]

    @staticmethod
    def _get_nodes(element: type[VertexElement], capacity: int, **values: object) -> GroupElement:
        """Return the variable group on the nodes of a growing network.

        Its seed nodes start at age 1.
        """
        return GroupElement(
            'nodes',
            element=element,
            on='sites',
            variable=True,
            capacity=capacity,
            values={'age': 1.0, **values},
            visual=DEGREE_VISUAL,
        )

    @staticmethod
    def _get_grown(
        nodes: GroupElement,
        growth: TopologyInteraction,
        space: NetworkSpace,
        capacity: int,
        interactions: Sequence[BaseInteraction] = (),
        *,
        label: str,
    ) -> CompositeSystem:
        """Return a growing network of a group on its nodes.

        The network has a variable edge group, the arrival of nodes, and its maps.
        """
        links = GroupElement('links', element=Element, on='edges', variable=True, capacity=capacity)
        return CompositeSystem(
            label,
            elements=[nodes, links],
            interactions=[growth, *BarabasiAlbertSystem._get_bookkeeping(), *interactions],
            space=space,
            time=DiscreteTime(),
        )

    def __init__(
        self,
        n_nodes: int = 303,
        *,
        edges: int = 2,
        seed_nodes: int = 3,
        fitness: bool = False,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.edges = edges
        self.seed_nodes = seed_nodes
        self.fitness = fitness
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return Barabási-Albert growth.

        By preferential attachment the degrees follow a power law `P(k) ~ k^-3`.
        """
        space = NetworkSpace(nx.complete_graph(self.seed_nodes))
        capacity = self.edges * self.n_nodes + self.seed_nodes**2
        if not self.fitness:
            nodes = BarabasiAlbertSystem._get_nodes(VertexElement, self.n_nodes)
            growth = TopologyInteraction(
                'growth',
                action='add-node',
                edges=self.edges,
                toward=BarabasiAlbertSystem._get_degree,
            )
            return BarabasiAlbertSystem._get_grown(nodes, growth, space, capacity, label=self.label)
        seed_fitness = np.random.default_rng(self.random_state).uniform(0.0, 1.0, self.seed_nodes)
        nodes = BarabasiAlbertSystem._get_nodes(FitnessVertexElement, self.n_nodes, fitness=seed_fitness.tolist())
        growth = TopologyInteraction(
            'growth',
            action='add-node',
            edges=self.edges,
            values={'fitness': lambda rng: rng.uniform(0.0, 1.0)},
            toward=lambda node: node.fitness * neighbors(1),
        )
        return BarabasiAlbertSystem._get_grown(nodes, growth, space, capacity, label=self.label)


class HolmeKimSystem(BaseSystem):
    """Holme-Kim growth.

    The network is scale-free like Barabási-Albert growth but with clustering that stays high.

    Every step a node joins with one edge by preferential attachment, and the next step it adds `edges - 1` edges, each
    to a neighbour of its first contact with probability `triad`, closing a triangle, and by preferential attachment
    otherwise. The system has the groups `nodes` and `links` and the arrival `growth`.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        triad:
            The probability that an added edge closes a triangle.

        seed_nodes:
            The number of nodes of the complete seed network.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import HolmeKimSystem
        >>> HolmeKimSystem(20).prepare().system_.elements[0].element.__name__
        'TriadVertexElement'
    """

    field = 'mathematics/networks'
    topics = ('network', 'growth', 'preferential attachment', 'clustering', 'triad formation')
    references = ('Holme and Kim 2002', 'Downey 2018')
    until = 300

    def __init__(
        self,
        n_nodes: int = 304,
        *,
        edges: int = 3,
        triad: float = 0.8,
        seed_nodes: int = 4,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.edges = edges
        self.triad = triad
        self.seed_nodes = seed_nodes

    @staticmethod
    def _get_triad_weight(node: SimpleNamespace) -> object:
        """Return the mix of triad formation and preferential attachment for a candidate.

        The candidate is a node.
        """
        closing = neighbors(node.other.near)
        attaching = neighbors(1)
        return node.triad * closing / (total(closing) + SMALLEST_TOTAL) + (1 - node.triad) * attaching / total(
            attaching,
        )

    def _build_system(self) -> CompositeSystem:
        """Return Holme-Kim growth.

        The network is scale-free like Barabási-Albert growth but with clustering that stays high.
        """
        space = NetworkSpace(nx.complete_graph(self.seed_nodes))
        nodes = BarabasiAlbertSystem._get_nodes(
            TriadVertexElement,
            self.n_nodes,
            triad=self.triad,
            extra=self.edges - 1,
        )
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
            toward=self._get_triad_weight,
            edges=self.edges - 1,
        )
        growth = TopologyInteraction('growth', action='add-node', toward=BarabasiAlbertSystem._get_degree)
        capacity = self.edges * self.n_nodes + self.seed_nodes**2
        return BarabasiAlbertSystem._get_grown(nodes, growth, space, capacity, [near, closure], label=self.label)


class VertexCopyingSystem(BaseSystem):
    """Vertex-copying growth.

    New nodes copy the edges of existing nodes and make a heavy-tailed degree.

    Every step a node joins with `edges` edges, each to a random node with probability `randomness` and otherwise to a
    random neighbour of a random prototype, which favours nodes of high degree without reading degrees. The system has
    the groups `nodes` and `links` and the arrival `growth`.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        randomness:
            The probability that an edge joins a random node instead of copying.

        seed_nodes:
            The number of nodes of the complete seed network.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import VertexCopyingSystem
        >>> VertexCopyingSystem(20).prepare().system_.interactions[0].edges
        3
    """

    field = 'mathematics/networks'
    topics = ('network', 'growth', 'copying', 'scale-free')
    references = ('Kleinberg et al. 1999', 'Krapivsky and Redner 2005', 'Ramasco 2020')
    until = 300

    def __init__(
        self,
        n_nodes: int = 304,
        *,
        edges: int = 3,
        randomness: float = 0.2,
        seed_nodes: int = 4,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.edges = edges
        self.randomness = randomness
        self.seed_nodes = seed_nodes

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        space = NetworkSpace(nx.complete_graph(self.seed_nodes))
        nodes = BarabasiAlbertSystem._get_nodes(CopyingVertexElement, self.n_nodes)
        growth = TopologyInteraction(
            'growth',
            action='add-node',
            edges=self.edges,
            toward=lambda node: self.randomness + (1 - self.randomness) * neighbors(node.other.inverse),
        )
        inverse = MapInteraction(
            'inverse',
            target='nodes',
            state='inverse',
            function=lambda s: 1 / (neighbors(1) + SMALLEST_DEGREE),
        )
        return BarabasiAlbertSystem._get_grown(
            nodes,
            growth,
            space,
            self.edges * self.n_nodes + self.seed_nodes**2,
            [inverse],
            label=self.label,
        )


class PopularitySimilarityNetworkSystem(BaseSystem):
    """Popularity-similarity growth, scale-free and clustered.

    Its nodes attach by hyperbolic distance.

    Every step a node joins at a uniform angle on the circle and the radius `2 ln t`, every other node `s` moves to the
    radius `b 2 ln s + (1 - b) 2 ln t` with `b = 1 / (exponent - 1)`, and the new node joins `edges` nodes drawn in
    proportion to the Fermi-Dirac probability of their hyperbolic distance `x = r_s + r_t + 2 ln(theta / 2)`, at the
    temperature `temperature`. The degrees follow a power law of the given exponent, and the clustering falls as the
    temperature rises (Papadopoulos et al. 2012). The system has the groups `nodes` and `links`, the arrival `growth`,
    and the states `nodes.angle`, `nodes.age`, and `nodes.degree`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The exponent is not above 2, or the temperature is not between 0 and 1.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import PopularitySimilarityNetworkSystem
        >>> system = PopularitySimilarityNetworkSystem(13).prepare().system_
        >>> trajectory = simulate(system, until=10, random_state=0)
        >>> int(trajectory.network(-1).sources.size)
        23
    """

    field = 'mathematics/networks'
    topics = ('network', 'growth', 'hyperbolic', 'popularity-similarity', 'scale-free', 'clustering')
    references = (
        'Papadopoulos et al. 2012, arXiv 1106.0286',
        'Gabrielli, Garlaschelli, Patil, and Serrano 2024, arXiv 2412.12988',
    )
    until = 297

    def __init__(
        self,
        n_nodes: int = 300,
        *,
        edges: int = 2,
        exponent: float = 2.5,
        temperature: float = 0.2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.edges = edges
        self.exponent = exponent
        self.temperature = temperature
        self.random_state = random_state

    @staticmethod
    def _get_similarity_weight(
        node: SimpleNamespace,
        arrivals: int,
        *,
        edges: int,
        fading: float,
        temperature: float,
    ) -> object:
        """Return the Fermi-Dirac probability of a candidate node's hyperbolic distance.

        The distance is that between the candidate node and the new node.
        """
        now = node.t + arrivals
        radius = 2 * ln(now)
        candidate = fading * 2 * ln(now - node.age) + (1 - fading) * radius
        angle = pi - abs(pi - abs(node.angle - node.source.angle))
        distance = candidate + radius + 2 * ln(angle / 2 + SMALLEST_ANGLE)
        spread = (
            2 * temperature * (1 - exp(-(1 - fading) * radius / 2)) / (sin(temperature * pi) * edges * (1 - fading))
        )
        return 1 - logistic(distance, radius - 2 * ln(spread), 2 * temperature)

    def _build_system(self) -> CompositeSystem:
        """Return popularity-similarity growth, scale-free and clustered.

        Its nodes attach by hyperbolic distance.
        """
        if self.exponent <= MIN_EXPONENT:
            error_msg = (
                f'The degree exponent of the growth is {self.exponent!r}. Give an exponent above {MIN_EXPONENT}.'
            )
            raise ValidationError(error_msg)
        if not 0 < self.temperature < 1:
            error_msg = f'The temperature of the growth is {self.temperature!r}. Give a temperature between 0 and 1.'
            raise ValidationError(error_msg)
        seed_nodes = self.edges + 1
        fading = 1 / (self.exponent - 1)
        angles = np.random.default_rng(self.random_state).uniform(0.0, 2 * np.pi, seed_nodes)
        ages = np.arange(seed_nodes - 1, -1, -1, dtype=float)
        nodes = BarabasiAlbertSystem._get_nodes(
            SimilarityVertexElement,
            self.n_nodes,
            age=ages.tolist(),
            angle=angles.tolist(),
        )
        growth = TopologyInteraction(
            'growth',
            action='add-node',
            edges=self.edges,
            values={'angle': lambda rng: rng.uniform(0.0, 2 * np.pi)},
            toward=lambda node: self._get_similarity_weight(
                node,
                seed_nodes + 1,
                edges=self.edges,
                fading=fading,
                temperature=self.temperature,
            ),
        )
        return BarabasiAlbertSystem._get_grown(
            nodes,
            growth,
            NetworkSpace(nx.complete_graph(seed_nodes)),
            self.edges * self.n_nodes + seed_nodes**2,
            label=self.label,
        )


class GraphletArrivalSystem(BaseSystem):
    """Growth by the arrival of three-node graphlets.

    The tree has a degree exponent that rises with `alpha`.

    The network starts from one edge, and every step a graphlet of a central node and two peripheral nodes joined to it
    arrives and merges one of its nodes with an existing node drawn with probability proportional to its degree: a
    peripheral node with probability `alpha`, so the existing node gains the central node, which carries the other
    peripheral node, and the central node otherwise, so the existing node gains both peripheral nodes (Filkov et al.
    2009, as Raman 2021, section 3.2.4.2, describes). Every step adds two nodes and two edges, and the shares of nodes
    of degree 1 and 2 are `2 (2 - alpha) / 5` and `alpha (7 - alpha) / 15`. The system has the groups `nodes` and
    `links` and the arrivals `growth` and `graphlet`.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        alpha:
            The probability that the graphlet attaches by a peripheral node.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import GraphletArrivalSystem
        >>> trajectory = simulate(GraphletArrivalSystem(22), until=10, random_state=0)
        >>> int(trajectory.network(-1).nodes.size), int(trajectory.network(-1).sources.size)
        (22, 21)
    """

    field = 'mathematics/networks'
    topics = ('network', 'growth', 'graphlet', 'preferential attachment', 'tree')
    references = ('Filkov et al. 2009', 'Raman 2021')
    until = 300

    def __init__(
        self,
        n_nodes: int = 602,
        *,
        alpha: float = 0.3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.alpha = alpha

    def _build_system(self) -> CompositeSystem:
        """Return growth by the arrival of three-node graphlets.

        The tree has a degree exponent that rises with `alpha`.
        """
        nodes = BarabasiAlbertSystem._get_nodes(VertexElement, self.n_nodes)
        growth = TopologyInteraction('growth', action='add-node', toward=BarabasiAlbertSystem._get_degree)
        graphlet = TopologyInteraction(
            'graphlet',
            action='add-node',
            values={'age': 0.0},
            toward=lambda node: self.alpha * where(node.age < JOINING_AGE, 1, 0)
            + (1 - self.alpha)
            * where(node.age > JOINING_AGE, 1, 0)
            * neighbors(where(node.other.age < JOINING_AGE, 1, 0)),
        )
        return BarabasiAlbertSystem._get_grown(
            nodes,
            growth,
            NetworkSpace(nx.path_graph(2)),
            self.n_nodes,
            [graphlet],
            label=self.label,
        )
