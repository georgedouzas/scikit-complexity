"""Random graph ensembles, Kronecker and configuration-model graphs, robustness.

The robustness is that of networks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import components, maximum, mod, neighbors, where
from ....core.components.spaces import AdjacencyNeighborhood, BaseSpace, LayerSpec, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.networks import NetworkNodeElement
from ..common.networks._regulated_node import ACTIVITY_VISUAL
from ..common.networks._vertex import DEGREE_VISUAL

ENSEMBLES = ('erdos_renyi', 'watts_strogatz')
STRATEGIES = ('random', 'targeted')


class DependentNodeElement(Element):
    """Nodes of two interdependent networks, each functional or not by its value `active`.

    A node stands for a pair of nodes, one of each network, that depend on each other.

    Args:
        label:
            The label of the nodes.

        active:
            Whether a node is functional, 1 or 0, one per node or shared.

        space:
            The network of two layers with one node per site, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import DependentNodeElement
        >>> DependentNodeElement('nodes', active=0.0).active
        0.0
    """

    visual = ACTIVITY_VISUAL

    def __init__(
        self,
        label: str,
        *,
        active: Annotated[object, FiniteEnumeratedSet([0, 1])] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.active = active


class InterdependenceInteraction(Interaction):
    """Failure of every node that leaves the giant component of the functional nodes of either network.

    The steps alternate between the two networks: even steps keep the nodes of the giant component of the graph of the
    network, and odd steps those of the giant component of the layer `layer`, among the nodes functional so far.

    Args:
        label:
            The label of the interaction.

        target:
            The nodes whose values `active` the cascade sets.

        layer:
            The name of the layer of the second network.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import InterdependenceInteraction
        >>> InterdependenceInteraction('cascade').layer
        'dependent'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'nodes', layer: str = 'dependent') -> None:
        super().__init__(label)
        self.target = target
        self.layer = layer

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the activity that survives the giant component of the network of the step as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        node = context.target
        both = node.active * node.other.active
        first = components(both, reduce='giant')
        second = components(both, over=AdjacencyNeighborhood(layer=self.layer), reduce='giant')
        self.laws_: list[BaseLaw] = [
            UpdateLaw(node.active, node.active * where(mod(context.t, 2) < 1, first, second)),
        ]


class EnsembleNodeElement(Element):
    """Node of a random graph, with its degree and its probability to add an edge.

    The edge goes to a random node, once per step.

    Args:
        label:
            The label of the node group.

        degree:
            The number of edges of the node, read at the start of every step.

        rate:
            The probability that the node adds an edge to a random node in a step.

        position:
            The place of the node on the ring of a Watts-Strogatz graph.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import EnsembleNodeElement
        >>> EnsembleNodeElement('nodes', rate=0.1).rate
        0.1
    """

    visual = DEGREE_VISUAL

    def __init__(
        self,
        label: str,
        *,
        degree: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        rate: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        position: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.degree = degree
        self.rate = rate
        self.position = position


class LatticeLinkElement(Element):
    """Link of a Watts-Strogatz graph, with the places of the two nodes it joined on the ring.

    A link whose ends still sit at these places has not been rewired.

    Args:
        label:
            The label of the link group.

        first:
            The place on the ring of the end of the link first in node order.

        second:
            The place on the ring of the other end of the link.

        space:
            The network whose links hold one member each, or `None` for one link.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import LatticeLinkElement
        >>> LatticeLinkElement('links', first=0.0, second=1.0).second
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        second: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space, on='edges' if space is not None else None)
        self.first = first
        self.second = second


class FailingElement(Element):
    """Nodes of a network under failures or attacks, each with its degree `degree`.

    Args:
        label:
            The label of the nodes.

        degree:
            The initial number of edges of every node, one per node or shared.

        space:
            The network with one node per site, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import FailingElement
        >>> FailingElement('nodes', degree=3.0).degree
        3.0
    """

    visual = DEGREE_VISUAL

    def __init__(
        self,
        label: str,
        *,
        degree: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.degree = degree


class AttackInteraction(Interaction):
    """Removal of the nodes of a network by random failures or by an attack on the nodes of the largest degree.

    The degree of every node is read from the edges at the start of every step. Under random failures every node fails
    with the probability `rate` per step, and under a targeted attack the nodes of the largest degree are removed, one
    degree class per step.

    Args:
        label:
            The label of the interaction.

        target:
            The nodes whose degrees the interaction sets and that fail.

        rate:
            The probability that a node fails in a step under random failures.

        targeted:
            Whether the nodes of the largest degree are attacked, in place of random failures.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import AttackInteraction
        >>> AttackInteraction('removal', targeted=False).targeted
        False
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'nodes', rate: float = 0.01, targeted: bool = True) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate
        self.targeted = targeted

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the degrees and the removals as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        degree = neighbors(1)
        removal = where(degree - maximum(degree) >= 0, 1, 0) if self.targeted else context.rate
        self.laws_: list[BaseLaw] = [
            UpdateLaw(context.target.degree, degree),
            UpdateLaw(TopologyRequestSpec('remove-node'), removal),
        ]


class NetworkAttackSystem(BaseSystem):
    """Failures or attacks on a scale-free network.

    The network is robust to random failures but fragile to attacks on hubs.

    The network is a Barabási-Albert graph. Under random failures every node fails with probability `rate` per step, and
    the giant component shrinks slowly. Under a targeted attack the nodes of the largest degree are removed every step,
    and the giant component breaks up after a few percent of the nodes are gone. The system has the groups `nodes` and
    `links`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The strategy is unknown.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NetworkAttackSystem
        >>> trajectory = NetworkAttackSystem(50).prepare().simulate(until=3, random_state=0)
        >>> int(trajectory.get_network(-1).nodes.size) < 50
        True
    """

    field = 'mathematics/networks'
    topics = ('network', 'robustness', 'percolation', 'attack', 'scale-free')
    references = ('Albert, Jeong, and Barabási 2000', 'Cohen et al. 2000', 'Ramasco 2020')
    until = 100

    def __init__(
        self,
        n_nodes: int = 300,
        *,
        edges: int = 2,
        strategy: str = 'targeted',
        rate: float = 0.01,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.edges = edges
        self.strategy = strategy
        self.rate = rate
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Barabási-Albert network and the removal of its nodes.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The strategy is unknown.
        """
        super()._prepare(context)
        if self.strategy not in STRATEGIES:
            error_msg = f'Unknown strategy {self.strategy!r}. Use one of {", ".join(STRATEGIES)}.'
            raise ValidationError(error_msg)
        space = NetworkSpace(nx.barabasi_albert_graph(self.n_nodes, self.edges, seed=self.random_state))
        removal = AttackInteraction('removal', target='nodes', rate=self.rate, targeted=self.strategy == 'targeted')
        self.system_ = System(self.label, elements=[FailingElement('nodes', space=space)], interactions=[removal])


class RandomGraphEnsemblesSystem(BaseSystem):
    """The construction of an Erdős-Rényi or a Watts-Strogatz random graph.

    The graph is built one step of new edges at a time.

    The Erdős-Rényi graph starts empty and every node adds an edge to a random node with a probability that brings the
    mean degree to `mean_degree` after `steps` steps, so a giant component emerges when the mean degree passes 1. A node
    adds at most one edge per step, so the construction takes at least `mean_degree / 2` steps. The Watts-Strogatz
    graph starts as a ring where every node `i` joins its `mean_degree` nearest nodes, and every edge from `i` to
    `i + j` not yet rewired keeps `i` and moves its other end to a random node at a rate that rewires a fraction
    `rewiring` of the edges after `steps` steps, as `nx.watts_strogatz_graph` does. Every edge is rewired at most once,
    so every node keeps at least `mean_degree / 2` edges, and paths shorten while the clustering stays high. The system
    has the groups `nodes` and `links`, the map `degree`, and the interaction `linking` or `rewiring`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The ensemble is unknown, or the Erdős-Rényi mean degree needs more than `steps` steps.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import RandomGraphEnsemblesSystem
        >>> system = RandomGraphEnsemblesSystem(40, ensemble='watts_strogatz', steps=5).prepare()
        >>> trajectory = system.simulate(until=5, random_state=0)
        >>> int(trajectory.get_network(-1).sources.size)
        80
    """

    field = 'mathematics/networks'
    topics = ('network', 'random graph', 'small world', 'giant component')
    references = ('Erdős and Rényi 1959', 'Watts and Strogatz 1998', 'Barabási 2016')
    until = 100

    def __init__(
        self,
        n_nodes: int = 200,
        *,
        ensemble: str = 'erdos_renyi',
        mean_degree: float = 4.0,
        rewiring: float = 0.2,
        steps: int = 100,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.ensemble = ensemble
        self.mean_degree = mean_degree
        self.rewiring = rewiring
        self.steps = steps

    @staticmethod
    def _get_degree() -> Interaction:
        """Return the update that sets the degree of every node from its edges."""
        return Interaction(
            'degree',
            roles={'target': 'nodes'},
            laws=lambda s: [UpdateLaw(s.target.degree, neighbors(1))],
        )

    @staticmethod
    def _get_variable_network(space: NetworkSpace, rate: float = 0.0) -> list[Element]:
        """Return the groups on the nodes and the edges of a network, which grow and shrink with it."""
        return [EnsembleNodeElement('nodes', rate=rate, space=space), Element('links', space=space, on='edges')]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The ensemble is unknown, or the Erdős-Rényi mean degree needs more than `steps` steps.
        """
        super()._prepare(context)
        if self.ensemble not in ENSEMBLES:
            error_msg = f'Unknown ensemble {self.ensemble!r}. Use one of {", ".join(ENSEMBLES)}.'
            raise ValidationError(error_msg)
        if self.ensemble == 'erdos_renyi':
            fewest = math.ceil(self.mean_degree / 2)
            if self.steps < fewest:
                error_msg = (
                    f'A node adds at most one edge per step, so a mean degree of {self.mean_degree!r} needs at least '
                    f'{fewest} steps, not {self.steps!r}. Give `steps` of at least {fewest}.'
                )
                raise ValidationError(error_msg)
            space = NetworkSpace(nx.empty_graph(self.n_nodes))
            rate = self.mean_degree / 2 / self.steps
            groups = RandomGraphEnsemblesSystem._get_variable_network(space, rate)
            linking = Interaction(
                'linking',
                roles={'target': 'nodes'},
                laws=lambda s: [UpdateLaw(TopologyRequestSpec('add-edge'), s.target.rate)],
            )
            self.system_ = System(
                self.label,
                elements=groups,
                interactions=[RandomGraphEnsemblesSystem._get_degree(), linking],
                space=space,
            )
            return
        ring = nx.watts_strogatz_graph(self.n_nodes, int(self.mean_degree), 0.0)
        space = NetworkSpace(ring)
        ends = np.sort(np.asarray(ring.edges(), dtype=float).reshape(-1, 2), axis=1)
        nodes = EnsembleNodeElement('nodes', position=np.arange(self.n_nodes, dtype=float), space=space)
        links = LatticeLinkElement('links', first=ends[:, 0], second=ends[:, 1], space=space)
        rate = 1 - (1 - self.rewiring) ** (1 / self.steps)
        reach = int(self.mean_degree) // 2
        rewiring = Interaction(
            'rewiring',
            among='nodes',
            along=True,
            laws=lambda s: self._get_rewiring(s, rate, reach),
        )
        self.system_ = System(
            self.label,
            elements=[nodes, links],
            interactions=[RandomGraphEnsemblesSystem._get_degree(), rewiring],
            space=space,
        )

    @staticmethod
    def _get_rewiring(edge: SimpleNamespace, rate: float, reach: int) -> list[BaseLaw]:
        """Return the rewiring of a link of the ring that has not been rewired yet.

        The link keeps its lattice end and moves the other one, as `nx.watts_strogatz_graph` does. The lattice end of
        the link from the place `i` to the place `i + j` is `i`, which comes last in node order when the link wraps
        around the ring. A rewired link has left the places it joined, so it is never rewired again.
        """
        moved = (edge.source.position - edge.edge.first) ** 2 + (edge.target.position - edge.edge.second) ** 2
        chance = rate * where(moved < 1, 1, 0)
        wraps = where(edge.edge.second - edge.edge.first > reach, 1, 0)
        return [
            UpdateLaw(TopologyRequestSpec('rewire'), chance * (1 - wraps)),
            UpdateLaw(TopologyRequestSpec('rewire', keep='target'), chance * wraps),
        ]


class KroneckerGraphSystem(BaseSystem):
    """The Kronecker graph of two graphs.

    Its adjacency matrix is the Kronecker product of theirs.

    The node `(i, j)` joins `(k, l)` when `i` joins `k` in the first graph and `j` joins `l` in the second (Gros 2015,
    exercise of chapter 1). The degree of `(i, j)` is the product of the degrees of `i` and `j`, so the mean degree is
    the product of the mean degrees, and the degree distribution is the distribution of the products. The system has the
    group `nodes` and their states `degree` and `clustering`.

    Args:
        first:
            The first graph, or `None` for a Barabási-Albert graph of 30 nodes with two edges per new node.

        second:
            The second graph, or `None` for a Barabási-Albert graph of 20 nodes with one edge per new node.

        random_state:
            The seed or generator of the default graphs.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.mathematics.networks import KroneckerGraphSystem
        >>> KroneckerGraphSystem(nx.path_graph(3), nx.path_graph(2)).prepare().system_.elements[0].size
        6
    """

    field = 'mathematics/networks'
    topics = ('network', 'kronecker product', 'random graph', 'degree distribution')
    references = ('Leskovec et al. 2010', 'Gros 2015')
    until = 1
    dt = 1

    @staticmethod
    def _get_static_system(label: str, graph: nx.Graph) -> System:
        """Return the system of a static network whose nodes hold their degree and clustering."""
        clustering = nx.clustering(graph)
        nodes = NetworkNodeElement(
            'nodes',
            degree=np.asarray([graph.degree(node) for node in graph], dtype=float),
            clustering=np.asarray([clustering[node] for node in graph], dtype=float),
            size=graph.number_of_nodes(),
        )
        return System(label, elements=[nodes], space=NetworkSpace(graph))

    def __init__(
        self,
        first: nx.Graph | None = None,
        second: nx.Graph | None = None,
        *,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.first = first
        self.second = second
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the network and its nodes."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        first = nx.barabasi_albert_graph(30, 2, seed=int(rng.integers(2**31))) if self.first is None else self.first
        second = nx.barabasi_albert_graph(20, 1, seed=int(rng.integers(2**31))) if self.second is None else self.second
        graph = nx.convert_node_labels_to_integers(nx.tensor_product(first, second), label_attribute='pair')
        self.system_ = KroneckerGraphSystem._get_static_system(self.label, graph)


class ConfigurationModelSystem(BaseSystem):
    """The configuration model.

    It is the maximally random network of a given degree sequence.

    Every node gets as many half-links as its degree, and pairs of half-links drawn uniformly at random become links
    (Jensen 2023, section 8.5.1, and Thurner et al. 2018, section 4.5.1). Without a degree sequence the degrees are
    drawn from the power law `P(k) ~ k^-exponent` for `k >= min_degree`, cut at `sqrt(n_nodes)`, and one half-link is
    added to a random node when their sum is odd. The pairing leaves self-loops and multilinks, which the rewiring of
    Thurner et al. removes while it keeps every degree. A neighbour has the mean degree `<k²> / <k>`, above the mean
    degree, the friendship paradox. The system has the group `nodes` and their states `degree` and `clustering`.

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
            The seed or generator of the degrees, the pairing, and the rewiring. They are drawn when the system is
            prepared from `np.random.default_rng(random_state)`, as in scikit-learn, so they follow the first draws of
            that generator. The `random_state` of `simulate` seeds the run apart.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The degree sum of a given sequence is odd, or rewiring cannot make the graph simple.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import ConfigurationModelSystem
        >>> system = ConfigurationModelSystem([3, 3, 2, 2, 2]).prepare().system_
        >>> system.elements[0].degree.tolist()
        [3.0, 3.0, 2.0, 2.0, 2.0]
    """

    field = 'mathematics/networks'
    topics = ('network', 'configuration model', 'random graph', 'degree sequence', 'friendship paradox')
    references = ('Molloy and Reed 1995', 'Jensen 2023', 'Thurner et al. 2018')
    until = 1
    dt = 1

    def __init__(
        self,
        degrees: Sequence[int] | None = None,
        *,
        n_nodes: int = 1000,
        exponent: float = 2.5,
        min_degree: int = 2,
        simple: bool = True,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.degrees = degrees
        self.n_nodes = n_nodes
        self.exponent = exponent
        self.min_degree = min_degree
        self.simple = simple
        self.random_state = random_state

    @staticmethod
    def _get_matched_stubs(degrees: npt.NDArray[np.int64], rng: np.random.Generator) -> list[tuple[int, int]]:
        """Return the links of a uniformly random pairing of the half-links of the nodes.

        The links may hold loops and multilinks.
        """
        stubs = rng.permutation(np.repeat(np.arange(degrees.size), degrees))
        return [(int(first), int(second)) for first, second in stubs.reshape(-1, 2)]

    @staticmethod
    def _get_simple_links(links: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
        """Return the links rewired until no self-loop or multilink is left.

        The rewiring keeps every degree.

        A self-loop or a repeated link and a link drawn at random are cut and their ends glued crosswise, as in the
        rewiring of Thurner et al. (2018, section 4.5.1.1).
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the network and its nodes."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        if self.degrees is None:
            support = np.arange(self.min_degree, max(self.min_degree, int(np.sqrt(self.n_nodes))) + 1)
            weights = support.astype(float) ** -self.exponent
            drawn = rng.choice(support, size=self.n_nodes, p=weights / weights.sum())
            drawn[int(rng.integers(self.n_nodes))] += int(drawn.sum() % 2)
        else:
            drawn = np.asarray(self.degrees, dtype=np.int64)
            if drawn.sum() % 2:
                error_msg = f'The degrees sum to {int(drawn.sum())}. Give a degree sequence with an even sum.'
                raise ValidationError(error_msg)
        links = self._get_matched_stubs(drawn, rng)
        links = self._get_simple_links(links, rng) if self.simple else [link for link in links if link[0] != link[1]]
        graph = nx.Graph()
        graph.add_nodes_from(range(drawn.size))
        graph.add_edges_from(links)
        self.system_ = KroneckerGraphSystem._get_static_system(self.label, graph)


class InterdependentNetworkCascadeSystem(BaseSystem):
    """The cascade of failures of Buldyrev between two interdependent networks.

    The cascade is that of Buldyrev et al., and the nodes of the networks depend on each other.

    Two Erdős-Rényi networks of the mean degree `mean_degree` share their nodes, every node of one depending on the node
    of the same index in the other, so a node is one pair of dependent nodes. An attack leaves every node functional
    with probability `p`. At even steps the nodes outside the giant component of the functional nodes of the first
    network fail, and at odd steps those outside that of the second network, until the cascade stops at the mutual giant
    component (Buldyrev et al. 2010, as Artime et al. 2025, section 5, Eqs. 9 to 11, review it). Its share `S` solves `S
    = p (1 - exp(-k S))²`, which has a solution above 0 only for `p` above `2.4554 / k`, where the networks collapse at
    once in a discontinuous transition. The system has the functional nodes `nodes.active` over the graph of the first
    network and the layer `dependent` of the second.

    Args:
        n_nodes:
            The number of nodes of every network.

        mean_degree:
            The mean degree `k` of every network.

        p:
            The share of nodes the attack leaves functional.

        random_state:
            The seed or generator of the networks and the attack.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import InterdependentNetworkCascadeSystem
        >>> system = InterdependentNetworkCascadeSystem(200, p=0.3).prepare()
        >>> int(system.simulate(until=20).get_values('nodes.active')[-1].sum())
        0
    """

    field = 'mathematics/networks'
    topics = ('network', 'interdependent networks', 'cascade', 'percolation', 'multilayer')
    references = ('Buldyrev et al. 2010', 'Artime et al. 2025')
    until = 40

    def __init__(
        self,
        n_nodes: int = 2000,
        *,
        mean_degree: float = 4.0,
        p: float = 0.8,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.mean_degree = mean_degree
        self.p = p
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the network and its nodes."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        probability = self.mean_degree / (self.n_nodes - 1)
        first = nx.gnp_random_graph(self.n_nodes, probability, seed=int(rng.integers(2**31)))
        second = LayerSpec('dependent', nx.gnp_random_graph(self.n_nodes, probability, seed=int(rng.integers(2**31))))
        functional = (rng.uniform(size=self.n_nodes) < self.p).astype(float)
        nodes = DependentNodeElement('nodes', active=functional, space=NetworkSpace(first, layers=[second]))
        cascade = InterdependenceInteraction('cascade', target='nodes', layer='dependent')
        self.system_ = System(self.label, elements=[nodes], interactions=[cascade])
