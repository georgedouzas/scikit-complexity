"""Random graph ensembles, Kronecker and configuration-model graphs, and the robustness of networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import MapInteraction, TopologyInteraction
from ....core.components.laws.utils import components, maximum, mod, neighbors, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import LayerSpec
from ..._catalog import register_system
from ._dependent_node import DependentNodeElement
from ._ensemble_node import EnsembleNodeElement
from ._failing import FailingElement
from ._link import LinkElement
from ._network_node import NetworkNodeElement
from ._vertex import DEGREE_VISUAL

ENSEMBLES = ('erdos_renyi', 'watts_strogatz')


STRATEGIES = ('random', 'targeted')


def _get_removal(s: SimpleNamespace) -> object:
    """Return the probability that a node is removed in a step, at random or as a node of the largest degree."""
    degree = neighbors(1)
    hub = where(s.source.targeted * (degree - maximum(degree)) >= 0, s.source.targeted, 0)
    return (1 - s.source.targeted) * s.source.rate + hub


def _get_degree() -> MapInteraction:
    """Return the map that sets the degree of every node from its edges at the start of a step."""
    return MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1))


def _variable_network(
    element: type[Element],
    space: NetworkSpace,
    capacity: int,
    **values: object,
) -> list[GroupElement]:
    """Return the variable groups on the nodes and the edges of a network."""
    nodes = GroupElement(
        'nodes',
        element=element,
        on='sites',
        variable=True,
        capacity=space.prepare().size_,
        values=values,
        visual=DEGREE_VISUAL,
    )
    links = GroupElement('links', element=LinkElement, on='edges', variable=True, capacity=capacity)
    return [nodes, links]


@register_system(
    field='mathematics/networks',
    tags=('network', 'robustness', 'percolation', 'attack', 'scale-free'),
    references=('Albert, Jeong, and Barabási 2000', 'Cohen et al. 2000', 'Ramasco 2020'),
    defaults={'until': 100},
)
def network_attack(
    n_nodes: int = 300,
    *,
    edges: int = 2,
    strategy: str = 'targeted',
    rate: float = 0.01,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return failures or attacks on a scale-free network, robust to random failures but fragile to attacks on hubs.

    The network is a Barabási-Albert graph. Under random failures every node fails with probability `rate` per step,
    and the giant component shrinks slowly. Under a targeted attack the nodes of the largest degree are removed
    every step, and the giant component breaks up after a few percent of the nodes are gone.

    Args:
        n_nodes:
            The number of nodes.

        edges:
            The number of edges every node of the Barabási-Albert graph joins with.

        strategy:
            `'random'` for random failures or `'targeted'` for attacks on the largest degree.

        rate:
            The probability that a node fails in a step under random failures.

        random_state:
            The seed or generator of the network.

    Returns:
        system:
            The system, with the groups `nodes` and `links`.

    Raises:
        ValidationError:
            The strategy is unknown.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import network_attack
        >>> trajectory = Simulation(network_attack(50), random_state=0).prepare().run(until=3).trajectory_
        >>> int(trajectory.network(-1).nodes.size) < 50
        True
    """
    if strategy not in STRATEGIES:
        error_msg = f'Unknown strategy {strategy!r}. Use one of {", ".join(STRATEGIES)}.'
        raise ValidationError(error_msg)
    space = NetworkSpace(nx.barabasi_albert_graph(n_nodes, edges, seed=random_state))
    targeted = float(strategy == 'targeted')
    groups = _variable_network(FailingElement, space, space.graph.number_of_edges(), rate=rate, targeted=targeted)
    removal = TopologyInteraction('removal', target='nodes', action='remove-node', probability=_get_removal)
    return System(elements=groups, interactions=[_get_degree(), removal], space=space, time=DiscreteTime())


@register_system(
    field='mathematics/networks',
    tags=('network', 'random graph', 'small world', 'giant component'),
    references=('Erdős and Rényi 1959', 'Watts and Strogatz 1998', 'Barabási 2016'),
    defaults={'until': 100},
)
def random_graph_ensembles(
    n_nodes: int = 200,
    *,
    ensemble: str = 'erdos_renyi',
    mean_degree: float = 4.0,
    rewiring: float = 0.2,
    steps: int = 100,
) -> System:
    """Return the construction of an Erdős-Rényi or a Watts-Strogatz random graph, one step of new edges at a time.

    The Erdős-Rényi graph starts empty and every node adds an edge to a random node with a probability that brings
    the mean degree to `mean_degree` after `steps` steps, so a giant component emerges when the mean degree passes 1.
    The Watts-Strogatz graph starts as a ring where every node joins its `mean_degree` nearest nodes, and every edge
    moves its target to a random node at a rate that rewires a fraction `rewiring` of the edges after `steps` steps,
    which shortens paths while the clustering stays high.

    Args:
        n_nodes:
            The number of nodes.

        ensemble:
            `'erdos_renyi'` or `'watts_strogatz'`.

        mean_degree:
            The mean degree of the final graph, an even number for the Watts-Strogatz ring.

        rewiring:
            The fraction of the edges of the Watts-Strogatz ring rewired after `steps` steps.

        steps:
            The number of steps of the construction.

    Returns:
        system:
            The system, with the groups `nodes` and `links`, the map `degree`, and the interaction `linking` or
            `rewiring`.

    Raises:
        ValidationError:
            The ensemble is unknown.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import random_graph_ensembles
        >>> system = random_graph_ensembles(40, ensemble='watts_strogatz', steps=5)
        >>> trajectory = Simulation(system, random_state=0).prepare().run(until=5).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        80
    """
    if ensemble not in ENSEMBLES:
        error_msg = f'Unknown ensemble {ensemble!r}. Use one of {", ".join(ENSEMBLES)}.'
        raise ValidationError(error_msg)
    if ensemble == 'erdos_renyi':
        space = NetworkSpace(nx.empty_graph(n_nodes))
        rate = mean_degree / 2 / steps
        groups = _variable_network(
            EnsembleNodeElement,
            space,
            int(np.ceil(1.5 * mean_degree * n_nodes / 2)) + 10,
            rate=rate,
        )
        linking = TopologyInteraction('linking', target='nodes', action='add-edge', probability=lambda s: s.source.rate)
        return System(elements=groups, interactions=[_get_degree(), linking], space=space, time=DiscreteTime())
    space = NetworkSpace(nx.watts_strogatz_graph(n_nodes, int(mean_degree), 0.0))
    rate = 1 - (1 - rewiring) ** (1 / steps)
    groups = _variable_network(EnsembleNodeElement, space, space.graph.number_of_edges())
    rewiring = TopologyInteraction('rewiring', target='nodes', action='rewire', probability=rate)
    return System(elements=groups, interactions=[_get_degree(), rewiring], space=space, time=DiscreteTime())


def _node_group(graph: nx.Graph) -> GroupElement:
    """Return the group on the nodes of a static network with the degree and the clustering of every node."""
    clustering = nx.clustering(graph)
    nodes = list(graph.nodes())
    return GroupElement(
        'nodes',
        element=NetworkNodeElement,
        size=len(nodes),
        values={
            'degree': np.asarray([graph.degree(node) for node in nodes], dtype=float),
            'clustering': np.asarray([clustering[node] for node in nodes], dtype=float),
        },
        location=nodes,
    )


@register_system(
    field='mathematics/networks',
    tags=('network', 'kronecker product', 'random graph', 'degree distribution'),
    references=('Leskovec et al. 2010', 'Gros 2015'),
    defaults={'until': 1, 'dt': 1},
)
def kronecker_graph(
    first: nx.Graph | None = None,
    second: nx.Graph | None = None,
    *,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Kronecker graph of two graphs, whose adjacency matrix is the Kronecker product of theirs.

    The node `(i, j)` joins `(k, l)` when `i` joins `k` in the first graph and `j` joins `l` in the second (Gros
    2015, exercise of chapter 1). The degree of `(i, j)` is the product of the degrees of `i` and `j`, so the mean
    degree is the product of the mean degrees, and the degree distribution is the distribution of the products.

    Args:
        first:
            The first graph, or `None` for a Barabási-Albert graph of 30 nodes with two edges per new node.

        second:
            The second graph, or `None` for a Barabási-Albert graph of 20 nodes with one edge per new node.

        random_state:
            The seed or generator of the default graphs.

    Returns:
        system:
            The system, with the group `nodes` and their states `degree` and `clustering`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.mathematics.networks import kronecker_graph
        >>> kronecker_graph(nx.path_graph(3), nx.path_graph(2)).elements[0].prepare().size_
        6
    """
    rng = np.random.default_rng(random_state)
    first = nx.barabasi_albert_graph(30, 2, seed=int(rng.integers(2**31))) if first is None else first
    second = nx.barabasi_albert_graph(20, 1, seed=int(rng.integers(2**31))) if second is None else second
    graph = nx.convert_node_labels_to_integers(nx.tensor_product(first, second), label_attribute='pair')
    return System(elements=[_node_group(graph)], space=NetworkSpace(graph))


def _matched_stubs(degrees: npt.NDArray[np.int64], rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return the links of a uniformly random pairing of the half-links of the nodes, with self-loops and multilinks."""
    stubs = rng.permutation(np.repeat(np.arange(degrees.size), degrees))
    return [(int(first), int(second)) for first, second in stubs.reshape(-1, 2)]


def _simple(links: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return the links rewired until no self-loop or multilink is left, keeping every degree.

    A self-loop or a repeated link and a link drawn at random are cut and their ends glued crosswise, as in the rewiring
    of Thurner et al. (2018, section 4.5.1.1).
    """
    links = [(min(link), max(link)) for link in links]
    for _ in range(100 * len(links)):
        seen: set[tuple[int, int]] = set()
        bad = []
        for index, link in enumerate(links):
            if link[0] == link[1] or link in seen:
                bad.append(index)
            seen.add(link)
        if not bad:
            return links
        first = bad[int(rng.integers(len(bad)))]
        second = int(rng.integers(len(links)))
        (a, b), (c, d) = links[first], links[second]
        crossed = [(min(a, c), max(a, c)), (min(b, d), max(b, d))]
        if first != second and all(x != y and (x, y) not in seen for x, y in crossed):
            links[first], links[second] = crossed[0], crossed[1]
    error_msg = 'The degree sequence admits no simple graph reachable by rewiring. Give a sparser degree sequence.'
    raise ValidationError(error_msg)


@register_system(
    field='mathematics/networks',
    tags=('network', 'configuration model', 'random graph', 'degree sequence', 'friendship paradox'),
    references=('Molloy and Reed 1995', 'Jensen 2023', 'Thurner et al. 2018'),
    defaults={'until': 1, 'dt': 1},
)
def configuration_model(
    degrees: Sequence[int] | None = None,
    *,
    n_nodes: int = 1000,
    exponent: float = 2.5,
    min_degree: int = 2,
    simple: bool = True,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the configuration model, the maximally random network of a given degree sequence.

    Every node gets as many half-links as its degree, and pairs of half-links drawn uniformly at random become links
    (Jensen 2023, section 8.5.1, and Thurner et al. 2018, section 4.5.1). Without a degree sequence the degrees are
    drawn from the power law `P(k) ~ k^-exponent` for `k >= min_degree`, cut at `sqrt(n_nodes)`, and one half-link
    is added to a random node when their sum is odd. The pairing leaves self-loops and multilinks, which the rewiring
    of Thurner et al. removes while it keeps every degree. A neighbour has the mean degree `<k²> / <k>`, above the
    mean degree, the friendship paradox.

    Args:
        degrees:
            The degree of every node, or `None` for degrees drawn from the power law.

        n_nodes:
            The number of nodes of the drawn degrees.

        exponent:
            The exponent of the power law of the drawn degrees.

        min_degree:
            The smallest drawn degree.

        simple:
            Whether self-loops and multilinks are rewired away, or dropped, which lowers some degrees.

        random_state:
            The seed or generator of the degrees, the pairing, and the rewiring.

    Returns:
        system:
            The system, with the group `nodes` and their states `degree` and `clustering`.

    Raises:
        ValidationError:
            The degree sum of a given sequence is odd, or rewiring cannot make the graph simple.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import configuration_model
        >>> system = configuration_model([3, 3, 2, 2, 2])
        >>> system.elements[0].values['degree'].tolist()
        [3.0, 3.0, 2.0, 2.0, 2.0]
    """
    rng = np.random.default_rng(random_state)
    if degrees is None:
        support = np.arange(min_degree, max(min_degree, int(np.sqrt(n_nodes))) + 1)
        weights = support.astype(float) ** -exponent
        drawn = rng.choice(support, size=n_nodes, p=weights / weights.sum())
        drawn[int(rng.integers(n_nodes))] += int(drawn.sum() % 2)
    else:
        drawn = np.asarray(degrees, dtype=np.int64)
        if drawn.sum() % 2:
            error_msg = f'The degrees sum to {int(drawn.sum())}. Give a degree sequence with an even sum.'
            raise ValidationError(error_msg)
    links = _matched_stubs(drawn, rng)
    links = _simple(links, rng) if simple else [link for link in links if link[0] != link[1]]
    graph = nx.Graph()
    graph.add_nodes_from(range(drawn.size))
    graph.add_edges_from(links)
    return System(elements=[_node_group(graph)], space=NetworkSpace(graph))


@register_system(
    field='mathematics/networks',
    tags=('network', 'interdependent networks', 'cascade', 'percolation', 'multilayer'),
    references=('Buldyrev et al. 2010', 'Artime et al. 2025'),
    defaults={'until': 40},
)
def interdependent_network_cascade(
    n_nodes: int = 2000,
    *,
    mean_degree: float = 4.0,
    p: float = 0.8,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the cascade of failures of Buldyrev et al. between two networks whose nodes depend on each other.

    Two Erdős-Rényi networks of the mean degree `mean_degree` share their nodes, every node of one depending on the
    node of the same index in the other, so a node is one pair of dependent nodes. An attack leaves every node
    functional with probability `p`. At even steps the nodes outside the giant component of the functional nodes of
    the first network fail, and at odd steps those outside that of the second network, until the cascade stops at the
    mutual giant component (Buldyrev et al. 2010, as Artime et al. 2025, section 5, Eqs. 9 to 11, review it). Its
    share `S` solves `S = p (1 - exp(-k S))²`, which has a solution above 0 only for `p` above `2.4554 / k`, where the
    networks collapse at once in a discontinuous transition.

    Args:
        n_nodes:
            The number of nodes of every network.

        mean_degree:
            The mean degree `k` of every network.

        p:
            The share of nodes the attack leaves functional.

        random_state:
            The seed or generator of the networks and the attack.

    Returns:
        system:
            The system, with the functional nodes `nodes.active` over the graph of the first network and the layer
            `dependent` of the second.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.networks import interdependent_network_cascade
        >>> system = interdependent_network_cascade(200, p=0.3)
        >>> int(Simulation(system).prepare().run(until=20).trajectory_.values('nodes.active')[-1].sum())
        0
    """
    rng = np.random.default_rng(random_state)
    probability = mean_degree / (n_nodes - 1)
    first = nx.gnp_random_graph(n_nodes, probability, seed=int(rng.integers(2**31)))
    second = LayerSpec('dependent', nx.gnp_random_graph(n_nodes, probability, seed=int(rng.integers(2**31))))
    space = NetworkSpace(first, layers=[second])
    functional = (rng.uniform(size=n_nodes) < p).astype(float)
    nodes = LatticeElement('nodes', element=DependentNodeElement, space=space, values={'active': functional})

    def _get_cascade(s: SimpleNamespace) -> object:
        """Return the activity of a node after the failures of the network of the step."""
        both = s.target.active * s.target.other.active
        first_giant = components(both, reduce='giant')
        second_giant = components(both, over=second, reduce='giant')
        return s.target.active * where(mod(s.t, 2) < 1, first_giant, second_giant)

    cascade = MapInteraction('cascade', target='nodes', state='active', function=_get_cascade)
    return System(elements=[nodes], interactions=[cascade], time=DiscreteTime())
