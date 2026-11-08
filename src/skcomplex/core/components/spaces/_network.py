"""Define network spaces.

They are graphs whose nodes members occupy, with named layers of relations over their nodes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.sparse
import xgi

from ...errors import SpaceError, ValidationError
from ...specs import LayerSpec, Tags
from ...types import Indices
from ._adjacency import AdjacencyNeighborhood
from ._base import BaseNeighborhood, BaseSpace

MIN_HYPEREDGE_NODES = 2


class NetworkSpace(BaseSpace):
    """Network whose nodes agents occupy or lattice sites sit on.

    The neighbours of a node are given by its edges. Lattice groups on a network hold one member per node, in the order
    of `graph.nodes`. The graph is the default layer, and `layers` adds named relations over the same nodes. Edges of a
    layer are numbered in the order of `graph.edges`, and every layer lowers to arrays: the source and target node index
    of every edge, and the adjacency of every node in compressed sparse row form. On an undirected graph the source of
    an edge is the end that comes first in node order. Build the graph with NetworkX, such as `nx.erdos_renyi_graph(100,
    0.1, seed=0)`.

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
            The hyperedges of the network, which hyperedge interactions and groups on hyperedges read: sequences of at
            least two nodes, an XGI `Hypergraph`, or an XGI `SimplicialComplex`, whose faces of at least two nodes are
            its hyperedges. The members of an XGI hyperedge follow the order of the nodes of the graph.

        label:
            The label of the space in its system.

    Attributes:
        graphs_:
            The graph of every layer by label, `None` first for the graph of the network.

        edges_:
            The source and target node index of every edge of every layer by label, in the order of `graph.edges`.

        adjacency_:
            The neighbours of every node of every layer by label in compressed sparse row form: the start of the
            neighbours of every node with the number of entries last, the sorted neighbours of every node in turn, the
            successors on a directed graph, and the edge that joins every node to every neighbour.

        hyperedges_:
            The node index of every member of every hyperedge, in the order of the hyperedges.

        incidence_:
            The sparse incidence matrix of the hyperedges, with one row per node and one column per hyperedge.

        memberships_:
            The indices of the hyperedges every node belongs to, in increasing order.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.components.spaces import NetworkSpace
        >>> from skcomplex.core.specs import LayerSpec
        >>> NetworkSpace(nx.cycle_graph(5)).prepare().size_
        5
        >>> space = NetworkSpace(nx.path_graph(3), layers=[LayerSpec('work', nx.complete_graph(3))])
        >>> space = space.prepare()
        >>> list(space.graphs_), space.graphs_['work'].number_of_edges()
        ([None, 'work'], 3)
        >>> [item.tolist() for item in space.edges_[None]]
        [[0, 1], [1, 2]]
        >>> [item.tolist() for item in space.adjacency_[None]]
        [[0, 1, 3, 4], [1, 0, 2, 1], [0, 0, 1, 1]]
        >>> import itertools
        >>> import xgi
        >>> triangles = xgi.Hypergraph([list(nodes) for nodes in itertools.combinations(range(5), 3)])
        >>> space = NetworkSpace(nx.complete_graph(5), hyperedges=triangles).prepare()
        >>> len(space.hyperedges_), space.hyperedges_[0], space.memberships_[0]
        (10, (0, 1, 2), (0, 1, 2, 3, 4, 5))
    """

    space_tags: ClassVar[tuple[str, ...]] = ('network', 'hypergraph')

    def _get_tags(self) -> Tags:
        """Return the tags of the network: a hypergraph with hyperedges, and no laws."""
        hypergraph = isinstance(self.hyperedges, xgi.Hypergraph) or len(self.hyperedges) > 0
        return Tags(space=frozenset({'hypergraph' if hypergraph else 'network'}))

    def __init__(
        self,
        graph: nx.Graph,
        *,
        layers: Sequence[LayerSpec] = (),
        capacity: int | None = None,
        fill: float = 0.0,
        hyperedges: Sequence[Sequence[object]] | xgi.Hypergraph = (),
        label: str = 'space',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.layers = layers
        self.capacity = capacity
        self.fill = fill
        self.hyperedges = hyperedges

    def _get_edges(self, graph: nx.Graph) -> tuple[Indices, Indices]:
        """Return the source and target node of every edge of a layer.

        If undirected, the end first in node order comes first.
        """
        index = {node: number for number, node in enumerate(self.graph.nodes)}
        pairs = np.array([(index[first], index[second]) for first, second in graph.edges()], dtype=np.intp)
        pairs = pairs.reshape(-1, 2)
        if not graph.is_directed():
            pairs = np.sort(pairs, axis=1)
        return pairs[:, 0].copy(), pairs[:, 1].copy()

    def _get_adjacency(self, sources: Indices, targets: Indices, directed: bool) -> tuple[Indices, Indices, Indices]:
        """Return the neighbours of every node in compressed sparse row form.

        It also returns the edge of every entry.
        """
        edges = np.arange(sources.size, dtype=np.intp)
        readers = sources if directed else np.concatenate([sources, targets])
        others = targets if directed else np.concatenate([targets, sources])
        ids = edges if directed else np.concatenate([edges, edges])
        order = np.lexsort((others, readers))
        counts = np.bincount(readers, minlength=self.size_)
        indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
        return indptr, others[order].astype(np.intp), ids[order].astype(np.intp)

    def _get_hyperedges(self) -> list[tuple[object, ...]]:
        """Return the nodes of every hyperedge.

        Those of an XGI hyperedge come in the order of the nodes of the graph.
        """
        if not isinstance(self.hyperedges, xgi.Hypergraph):
            return [tuple(hyperedge) for hyperedge in self.hyperedges]
        order = {node: number for number, node in enumerate(self.graph.nodes)}
        members = self.hyperedges.edges.members(dtype=list)
        return [
            tuple(sorted(hyperedge, key=lambda node: order.get(node, len(order))))
            for hyperedge in members
            if len(hyperedge) >= MIN_HYPEREDGE_NODES
        ]

    def _count_parts(self, on: str, layer: str | None = None) -> int:
        """Return the number of nodes, of edges of a layer, or of hyperedges.

        A group has one member on each.

        Args:
            on:
                `'sites'`, `'edges'`, or `'hyperedges'`.

            layer:
                The layer whose edges a group lives on, or `None` for the graph of the network.

        Returns:
            count:
                The number of parts.

        Raises:
            SpaceError:
                The network has no such layer, or no hyperedges.
        """
        if on == 'edges':
            if layer not in self.edges_:
                known = ', '.join(repr(label) for label in self.edges_)
                error_msg = f'The network has no layer {layer!r}. Its layers are {known}.'
                raise SpaceError(error_msg)
            return int(self.edges_[layer][0].size)
        if on == 'hyperedges' and self.hyperedges_:
            return len(self.hyperedges_)
        return super()._count_parts(on, layer)

    def _check(self) -> None:
        """Raise an error for a hyperedge or a layer that does not fit the graph nodes."""
        nodes = set(self.graph.nodes)
        for hyperedge in self._get_hyperedges():
            if len(hyperedge) < MIN_HYPEREDGE_NODES or not set(hyperedge) <= nodes:
                error_msg = f'The hyperedge {tuple(hyperedge)} needs at least two nodes of the graph.'
                raise ValidationError(error_msg)
        for layer in self.layers:
            layer._check()
        labels = [layer.label for layer in self.layers]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            error_msg = (
                f'The layers {", ".join(map(repr, repeated))} appear more than once. Give every layer its own label.'
            )
            raise ValidationError(error_msg)
        for layer in self.layers:
            if set(layer.graph.nodes) != nodes:
                error_msg = f'The layer {layer.label!r} has other nodes than the graph. Give it the nodes of the graph.'
                raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Set the graph, the edges, the adjacency of every layer, and the node positions."""
        self.size_ = int(self.graph.number_of_nodes())
        self.n_dim_ = 1
        self.shape_ = (self.size_,)
        self.periodic_ = False
        self.default_visual_ = 'sites'
        graphs: dict[str | None, nx.Graph] = {None: self.graph}
        graphs.update((layer.label, layer.graph) for layer in self.layers)
        self.graphs_ = graphs
        self.edges_ = {label: self._get_edges(graph) for label, graph in self.graphs_.items()}
        self.adjacency_ = {
            label: self._get_adjacency(*self.edges_[label], graph.is_directed())
            for label, graph in self.graphs_.items()
        }
        given = nx.get_node_attributes(self.graph, 'pos')
        if self.size_ and len(given) == self.size_:
            self.positions_ = np.asarray([given[node] for node in self.graph.nodes], dtype=float)
        else:
            layout = nx.circular_layout(self.graph)
            self.positions_ = np.asarray([layout[node] for node in self.graph.nodes], dtype=float).reshape(-1, 2)
        index = {node: number for number, node in enumerate(self.graph.nodes)}
        self.hyperedges_ = tuple(tuple(index[node] for node in hyperedge) for hyperedge in self._get_hyperedges())
        rows = [node for hyperedge in self.hyperedges_ for node in hyperedge]
        columns = [number for number, hyperedge in enumerate(self.hyperedges_) for _ in hyperedge]
        shape = (self.size_, len(self.hyperedges_))
        incidence = scipy.sparse.csr_array((np.ones(len(rows)), (rows, columns)), shape=shape)
        incidence.sort_indices()
        self.incidence_ = incidence
        self.memberships_ = tuple(
            tuple(int(item) for item in self.incidence_.indices[start:stop])
            for start, stop in zip(self.incidence_.indptr[:-1], self.incidence_.indptr[1:], strict=True)
        )

    def _default_neighborhood(self) -> AdjacencyNeighborhood:
        """Return the neighbours along the edges of the graph."""
        return AdjacencyNeighborhood()

    def _index_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the node index of every neighbour of every node.

        It is padded with the number of nodes. The table has one row per node and one column per neighbour slot, as wide
        as the largest degree, plus one column for the node itself when the neighbourhood includes it.
        """
        if not isinstance(neighborhood, AdjacencyNeighborhood):
            error_msg = f'A network reads an adjacency neighbourhood, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        size = self.size_
        indptr, indices, _ = self.adjacency_[neighborhood.layer]
        degrees = np.diff(indptr)
        table = np.full((size, int(degrees.max(initial=0))), size, dtype=np.intp)
        slots = np.arange(indices.size) - np.repeat(indptr[:-1], degrees)
        table[np.repeat(np.arange(size), degrees), slots] = indices
        if neighborhood.center:
            return np.column_stack([np.arange(size, dtype=np.intp), table])
        return table

    def _neighbor_pairs(self, layer: str | None = None) -> list[tuple[int, int]]:
        """Return the pairs of distinct nodes joined by an edge of a layer.

        They come in the order of its edges.
        """
        sources, targets = self.edges_[layer]
        return [
            (first, second) for first, second in zip(sources.tolist(), targets.tolist(), strict=True) if first != second
        ]
