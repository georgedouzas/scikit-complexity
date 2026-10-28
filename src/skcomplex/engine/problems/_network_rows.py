"""Variable groups of one network: the groups on its nodes and the edge group of every changing layer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...specs import BaseSpec
from ...types import Array, Indices, Pair
from ._capacity_rows import CapacityRows
from ._network_snapshot import NetworkSnapshot


@dataclass(frozen=True)
class NetworkRows(BaseSpec):
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
