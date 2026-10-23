"""Define network spaces, their relation layers, the graph neighbourhood, and the generators of networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..base import Requires, SpaceKind, ValidationError
from ._spaces import Space

type Seed = int | np.random.Generator | None
type Indices = npt.NDArray[np.intp]


def _simple(graph: nx.Graph) -> nx.Graph:
    """Return a simple graph without parallel edges and self-loops."""
    simple = nx.DiGraph(graph) if graph.is_directed() else nx.Graph(graph)
    simple.remove_edges_from(list(nx.selfloop_edges(simple)))
    return simple


@dataclass(frozen=True)
class Adjacency:
    """Neighbourhood of a node of a graph: the nodes along its edges, in node order, after the node itself if included.

    Rules receive the values of the neighbours padded to the largest degree of the graph. On a directed graph the
    neighbours are the successors.

    Args:
        center:
            Whether the node itself comes first.

        layer:
            The label of the layer whose edges join the neighbours, or `None` for the graph of the network.

    Examples:
        >>> from skcomplex.core.space import Adjacency
        >>> Adjacency(center=True).center
        True
        >>> Adjacency(layer='work').layer
        'work'
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.NETWORK})

    center: bool = False
    layer: str | None = None


@dataclass(frozen=True)
class Layer:
    """Relation over the nodes of a network space, such as contacts or employment, held as a graph of its own.

    A network space holds several layers over the same nodes. Edge groups and interactions over neighbours name the
    layer they bind to, and reductions reduce over its edges with `Adjacency(layer=label)` or the layer itself.

    Args:
        label:
            The name of the layer, unique in its network space.

        graph:
            The NetworkX graph of the relation, over the nodes of the network space.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.space import Layer
        >>> Layer('work', nx.path_graph(3)).label
        'work'
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.NETWORK})

    label: str
    graph: nx.Graph

    def __repr__(self) -> str:
        """Return the label and the size of the graph."""
        return f'Layer({self.label!r}, nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})'


class NetworkSpace(Space):
    """Network whose nodes agents occupy or lattice sites sit on, with the neighbours of a node given by its edges.

    Lattice groups on a network hold one member per node, in the order of `graph.nodes`. The graph is the default
    layer, and `layers` adds named relations over the same nodes. Edges of a layer are numbered in the order of
    `graph.edges`, and every layer lowers to arrays: the source and target node index of every edge, and the
    adjacency of every node in compressed sparse row form. On an undirected graph the source of an edge is the end
    that comes first in node order.

    Args:
        graph:
            The NetworkX graph, the default layer.

        layers:
            The named layers over the nodes of the graph.

        capacity:
            The largest number of agents on a node, or `None` for no limit.

        fill:
            The value lattice rules read in the padding slots of nodes with fewer neighbours than the largest degree.

        hyperedges:
            The hyperedges of the network, each a sequence of at least two nodes, which interactions with the binding
            `'hyperedges'` and groups on hyperedges read.

    Raises:
        ValidationError:
            A hyperedge has fewer than two nodes or a node outside the graph, or a layer has other nodes than the
            graph or a label that another layer has.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.space import Layer, NetworkSpace
        >>> NetworkSpace(nx.cycle_graph(5)).graph.number_of_nodes()
        5
        >>> NetworkSpace(nx.complete_graph(3), hyperedges=[(0, 1, 2)]).hyperedges
        [(0, 1, 2)]
        >>> space = NetworkSpace(nx.path_graph(3), layers=[Layer('work', nx.complete_graph(3))])
        >>> space.layer_labels, space.resolve_graph('work').number_of_edges()
        ((None, 'work'), 3)
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.NETWORK

    n_dim = 1

    def __init__(
        self,
        graph: nx.Graph,
        *,
        layers: Sequence[Layer] = (),
        capacity: int | None = None,
        fill: float = 0.0,
        hyperedges: Sequence[Sequence[object]] = (),
    ) -> None:
        self.graph = graph
        self.layers = layers
        self.capacity = capacity
        self.fill = fill
        self.hyperedges = hyperedges
        nodes = set(graph.nodes)
        for hyperedge in hyperedges:
            if len(hyperedge) < 2 or not set(hyperedge) <= nodes:  # noqa: PLR2004
                error_msg = f'The hyperedge {tuple(hyperedge)} needs at least two nodes of the graph.'
                raise ValidationError(error_msg)
        labels = [layer.label for layer in layers]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            listed = ', '.join(map(repr, repeated))
            error_msg = f'The layers {listed} appear more than once. Give every layer its own label.'
            raise ValidationError(error_msg)
        for layer in layers:
            if set(layer.graph.nodes) != nodes:
                error_msg = f'The layer {layer.label!r} has other nodes than the graph. Give it the nodes of the graph.'
                raise ValidationError(error_msg)

    @property
    def layer_labels(self) -> tuple[str | None, ...]:
        """Labels of the layers, `None` first for the graph itself."""
        return (None, *(layer.label for layer in self.layers))

    @property
    def size(self) -> int:
        """Number of nodes."""
        return int(self.graph.number_of_nodes())

    def resolve_graph(self, layer: str | None = None) -> nx.Graph:
        """Return the graph of a layer.

        Args:
            layer:
                The label of the layer, or `None` for the graph of the network.

        Returns:
            graph:
                The NetworkX graph of the layer.

        Raises:
            ValidationError:
                The network has no layer of that label.

        Examples:
            >>> import networkx as nx
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace(nx.path_graph(4)).resolve_graph().number_of_edges()
            3
        """
        if layer is None:
            return self.graph
        for item in self.layers:
            if item.label == layer:
                return item.graph
        known = ', '.join(repr(label) for label in self.layer_labels)
        error_msg = f'The network has no layer {layer!r}. Its layers are {known}.'
        raise ValidationError(error_msg)

    def list_edges(self, layer: str | None = None) -> tuple[Indices, Indices]:
        """Return the source and target node index of every edge of a layer, in the order of `graph.edges`.

        Args:
            layer:
                The label of the layer, or `None` for the graph of the network.

        Returns:
            sources:
                The node index of the source of every edge, the end first in node order on an undirected graph.

            targets:
                The node index of the target of every edge.

        Examples:
            >>> import networkx as nx
            >>> from skcomplex.core.space import NetworkSpace
            >>> [item.tolist() for item in NetworkSpace(nx.path_graph(3)).list_edges()]
            [[0, 1], [1, 2]]
        """
        graph = self.resolve_graph(layer)
        index = {node: number for number, node in enumerate(self.graph.nodes)}
        pairs = np.array([(index[first], index[second]) for first, second in graph.edges()], dtype=np.intp)
        pairs = pairs.reshape(-1, 2)
        if not graph.is_directed():
            pairs = np.sort(pairs, axis=1)
        return pairs[:, 0].copy(), pairs[:, 1].copy()

    def compute_adjacency(self, layer: str | None = None) -> tuple[Indices, Indices, Indices]:
        """Return the neighbours of every node of a layer in compressed sparse row form, with the edge of every entry.

        Args:
            layer:
                The label of the layer, or `None` for the graph of the network.

        Returns:
            indptr:
                The start of the neighbours of every node, and the number of entries last.

            indices:
                The neighbours of every node in turn, sorted, the successors on a directed graph.

            edges:
                The edge, in the order of `list_edges`, that joins every node to every neighbour.

        Examples:
            >>> import networkx as nx
            >>> from skcomplex.core.space import NetworkSpace
            >>> [item.tolist() for item in NetworkSpace(nx.path_graph(3)).compute_adjacency()]
            [[0, 1, 3, 4], [1, 0, 2, 1], [0, 0, 1, 1]]
        """
        sources, targets = self.list_edges(layer)
        edges = np.arange(sources.size, dtype=np.intp)
        directed = self.resolve_graph(layer).is_directed()
        readers = sources if directed else np.concatenate([sources, targets])
        others = targets if directed else np.concatenate([targets, sources])
        ids = edges if directed else np.concatenate([edges, edges])
        order = np.lexsort((others, readers))
        counts = np.bincount(readers, minlength=self.size)
        indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
        return indptr, others[order].astype(np.intp), ids[order].astype(np.intp)

    def read_positions(self, attribute: str = 'pos') -> npt.NDArray[np.float64]:
        """Return the position of every node from a node attribute, such as the positions of a spatial generator.

        Args:
            attribute:
                The name of the node attribute that holds the position.

        Returns:
            positions:
                One row of coordinates per node, in node order.

        Raises:
            ValidationError:
                A node has no such attribute.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.random_geometric(5, 0.5, seed=1).read_positions().shape
            (5, 2)
        """
        missing = [node for node, data in self.graph.nodes(data=True) if attribute not in data]
        if missing:
            error_msg = f'The nodes {missing[:3]} have no attribute {attribute!r}. Store a position on every node.'
            raise ValidationError(error_msg)
        return np.asarray([self.graph.nodes[node][attribute] for node in self.graph.nodes], dtype=float)

    def index_table(self, neighborhood: Adjacency) -> npt.NDArray[np.intp]:
        """Return the node index of every neighbour of every node, padded with the number of nodes.

        Args:
            neighborhood:
                The graph neighbourhood.

        Returns:
            table:
                One row per node and one column per neighbour slot, as wide as the largest degree, plus one column
                for the node itself when the neighbourhood includes it.

        Raises:
            ValidationError:
                The neighbourhood is not a graph neighbourhood.

        Examples:
            >>> import networkx as nx
            >>> from skcomplex.core.space import Adjacency, NetworkSpace
            >>> NetworkSpace(nx.star_graph(2)).index_table(Adjacency()).tolist()
            [[1, 2], [0, 3], [0, 3]]
        """
        if not isinstance(neighborhood, Adjacency):
            error_msg = f'A network reads the neighbourhood Adjacency(), but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        size = self.size
        indptr, indices, _ = self.compute_adjacency(neighborhood.layer)
        degrees = np.diff(indptr)
        width = int(degrees.max(initial=0))
        table = np.full((size, width), size, dtype=np.intp)
        slots = np.arange(indices.size) - np.repeat(indptr[:-1], degrees)
        table[np.repeat(np.arange(size), degrees), slots] = indices
        if neighborhood.center:
            return np.column_stack([np.arange(size, dtype=np.intp), table])
        return table

    @classmethod
    def erdos_renyi(
        cls,
        n: int,
        p: float,
        *,
        directed: bool = False,
        seed: Seed = None,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return a network of `n` nodes with every edge present with probability `p`, the Erdős-Rényi graph `G(n, p)`.

        Args:
            n:
                The number of nodes.

            p:
                The probability of every edge.

            directed:
                Whether the edges are directed.

            seed:
                The seed or the random number generator, such as one drawn from the simulation's random stream.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.erdos_renyi(100, 0.05, seed=1).graph.number_of_edges()
            267
        """
        return cls(nx.gnp_random_graph(n, p, seed=seed, directed=directed), layers=layers)

    @classmethod
    def watts_strogatz(
        cls,
        n: int,
        k: int,
        p: float,
        *,
        seed: Seed = None,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return a small-world network, a ring of nodes joined to their `k` nearest nodes with edges rewired at `p`.

        Args:
            n:
                The number of nodes.

            k:
                The number of nearest nodes in the ring every node joins, an even number.

            p:
                The probability of rewiring every edge.

            seed:
                The seed or the random number generator.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.watts_strogatz(20, 4, 0.1, seed=1).graph.number_of_edges()
            40
        """
        return cls(nx.watts_strogatz_graph(n, k, p, seed=seed), layers=layers)

    @classmethod
    def barabasi_albert(
        cls,
        n: int,
        m: int,
        *,
        seed: Seed = None,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return a scale-free network grown by preferential attachment of every new node with `m` edges.

        Args:
            n:
                The number of nodes.

            m:
                The number of edges of every new node.

            seed:
                The seed or the random number generator.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.barabasi_albert(50, 2, seed=1).graph.number_of_edges()
            96
        """
        return cls(nx.barabasi_albert_graph(n, m, seed=seed), layers=layers)

    @classmethod
    def configuration_model(
        cls,
        degrees: Sequence[int],
        *,
        seed: Seed = None,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return a random network with given degrees, without the parallel edges and self-loops the pairing draws.

        Args:
            degrees:
                The degree of every node, with an even sum.

            seed:
                The seed or the random number generator.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.configuration_model([3, 3, 2, 2, 2], seed=1).graph.number_of_nodes()
            5
        """
        return cls(_simple(nx.configuration_model(list(degrees), seed=seed)), layers=layers)

    @classmethod
    def stochastic_block_model(
        cls,
        sizes: Sequence[int],
        probabilities: Sequence[Sequence[float]],
        *,
        seed: Seed = None,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return a network of blocks whose nodes join with a probability that depends on their two blocks.

        Every node stores its block in the node attribute `block`.

        Args:
            sizes:
                The number of nodes of every block.

            probabilities:
                The probability of an edge between a node of block `a` and one of block `b`, a symmetric matrix.

            seed:
                The seed or the random number generator.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> space = NetworkSpace.stochastic_block_model([5, 5], [[0.9, 0.0], [0.0, 0.9]], seed=1)
            >>> space.graph.nodes[7]['block']
            1
        """
        graph = nx.stochastic_block_model(list(sizes), [list(row) for row in probabilities], seed=seed)
        return cls(graph, layers=layers)

    @classmethod
    def grid(
        cls,
        shape: Sequence[int],
        *,
        periodic: bool = False,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return the network of a square lattice of a shape, whose nodes are numbered in row-major order.

        Every node stores its lattice coordinates in the node attribute `pos`.

        Args:
            shape:
                The number of nodes along every dimension.

            periodic:
                Whether opposite sides join.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.grid((3, 3), periodic=True).graph.number_of_edges()
            18
        """
        grid = nx.grid_graph(list(reversed(list(shape))), periodic=periodic)
        cells = sorted(grid.nodes)
        mapping = {cell: index for index, cell in enumerate(cells)}
        graph = nx.Graph()
        positions = {index: tuple(float(item) for item in cell) for cell, index in mapping.items()}
        graph.add_nodes_from((index, {'pos': position}) for index, position in positions.items())
        graph.add_edges_from((mapping[first], mapping[second]) for first, second in grid.edges)
        return cls(graph, layers=layers)

    @classmethod
    def complete(
        cls,
        n: int,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return the complete network of `n` nodes, where every node joins every other.

        Args:
            n:
                The number of nodes.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.complete(5).graph.number_of_edges()
            10
        """
        return cls(nx.complete_graph(n), layers=layers)

    @classmethod
    def waxman(
        cls,
        n: int,
        *,
        beta: float = 0.4,
        alpha: float = 0.1,
        domain: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        seed: Seed = None,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return a spatial network of nodes placed at random, joined with a probability that decays with distance.

        Two nodes at distance `d` join with probability `beta exp(-d / (alpha L))`, where `L` is the largest distance
        in the domain. Every node stores its position in the node attribute `pos`, which `read_positions` returns.

        Args:
            n:
                The number of nodes.

            beta:
                The probability of an edge at zero distance.

            alpha:
                The decay length relative to the largest distance.

            domain:
                The rectangle `(x_min, y_min, x_max, y_max)` the nodes are placed in.

            seed:
                The seed or the random number generator.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.waxman(20, seed=1).read_positions().shape
            (20, 2)
        """
        return cls(nx.waxman_graph(n, beta=beta, alpha=alpha, domain=domain, seed=seed), layers=layers)

    @classmethod
    def random_geometric(
        cls,
        n: int,
        radius: float,
        *,
        dim: int = 2,
        seed: Seed = None,
        layers: Sequence[Layer] = (),
    ) -> 'NetworkSpace':
        """Return a spatial network of nodes placed at random in the unit cube, joined within a distance.

        Every node stores its position in the node attribute `pos`, which `read_positions` returns.

        Args:
            n:
                The number of nodes.

            radius:
                The largest distance of joined nodes.

            dim:
                The dimension of the cube.

            seed:
                The seed or the random number generator.

            layers:
                The named layers over the nodes of the network.

        Returns:
            space:
                The network space.

        Examples:
            >>> from skcomplex.core.space import NetworkSpace
            >>> NetworkSpace.random_geometric(30, 0.3, seed=1).graph.number_of_nodes()
            30
        """
        return cls(nx.random_geometric_graph(n, radius, dim=dim, seed=seed), layers=layers)
