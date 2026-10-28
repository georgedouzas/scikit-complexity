"""Variable groups of one network: the groups on its nodes and the edge group of every changing layer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...types import Array, Indices, Pair
from ..problems import NetworkSnapshot
from ._capacity import CapacityRows


@dataclass(frozen=True)
class NetworkRows:
    """Variable groups of one network: the groups on its nodes and the edge group of every changing layer.

    Args:
        nodes:
            The groups on the nodes, which share their rows.

        edges:
            The variable edge group of every layer by label, `None` for the graph of the network.

        directed:
            Whether the edges of every layer are directed.
    """

    nodes: tuple[CapacityRows, ...]
    edges: Mapping[str | None, CapacityRows]
    directed: Mapping[str | None, bool]

    def present(self, p: Array) -> npt.NDArray[np.bool_]:
        """Return whether every node row holds a node.

        Args:
            p:
                The parameter vector.

        Returns:
            present:
                One flag per node row.
        """
        return self.nodes[0].present(p)

    def order_ends(self, layer: str | None, first: int, second: int) -> Pair:
        """Return the ends of an edge, in node order on an undirected layer.

        Args:
            layer:
                The layer.

            first:
                The node row of one end.

            second:
                The node row of the other end.

        Returns:
            ends:
                The source and target node rows.
        """
        return (first, second) if self.directed[layer] else (min(first, second), max(first, second))

    def neighbours(self, layer: str | None, node: int, p: Array) -> Indices:
        """Return the node rows a node joins in a layer, its successors on a directed layer.

        Args:
            layer:
                The layer.

            node:
                The node row.

            p:
                The parameter vector.

        Returns:
            neighbours:
                The node rows.
        """
        edges = self.edges[layer]
        sources, targets = edges.ends(p)
        present = edges.present(p)
        found = targets[present & (sources == node)]
        if self.directed[layer]:
            return found
        return np.concatenate([found, sources[present & (targets == node)]])

    def add_node(self, x: Array, p: Array, values: Mapping[str, object]) -> int:
        """Add a node in the first free node row of every group on the nodes.

        Args:
            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

            values:
                The values of the new node by variable name.

        Returns:
            row:
                The node row.
        """
        row = self.nodes[0].take_row(p)
        for group in self.nodes:
            group.write(row, x, p, values)
        return row

    def add_edge(self, layer: str | None, ends: Pair, x: Array, p: Array, values: Mapping[str, object]) -> None:
        """Add an edge between two node rows to a layer.

        Args:
            layer:
                The layer.

            ends:
                The source and target node rows.

            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

            values:
                The values of the new edge by variable name.
        """
        edges = self.edges[layer]
        row = edges.take_row(p)
        first, second = self.order_ends(layer, *ends)
        edges.write(row, x, p, values)
        p[edges.source.start + row] = first  # type: ignore[union-attr]
        p[edges.target.start + row] = second  # type: ignore[union-attr]

    def remove_node(self, row: int, p: Array) -> None:
        """Remove a node and every edge that joins it.

        Args:
            row:
                The node row.

            p:
                The parameter vector, changed in place.
        """
        for group in self.nodes:
            group.clear(row, p)
        for edges in self.edges.values():
            sources, targets = edges.ends(p)
            for edge in np.flatnonzero(edges.present(p) & ((sources == row) | (targets == row))):
                edges.clear(int(edge), p)

    def snapshot(self, p: Array, layer: str | None) -> NetworkSnapshot:
        """Return the nodes and the edges of a layer present at the parameters of a frame.

        Args:
            p:
                The parameter vector of the frame.

            layer:
                The layer.

        Returns:
            snapshot:
                The nodes and edges present.
        """
        edges = self.edges[layer]
        present = edges.present(p)
        sources, targets = edges.ends(p)
        return NetworkSnapshot(
            np.flatnonzero(self.present(p)).astype(np.intp),
            sources[present],
            targets[present],
            np.flatnonzero(present).astype(np.intp),
        )
