"""Define the network snapshot spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class NetworkSnapshotSpec(BaseSpec):
    """Nodes and edges of a network layer at one frame.

    Args:
        nodes:
            The rows of the nodes present, in increasing order.

        sources:
            The node row of the source of every edge present.

        targets:
            The node row of the target of every edge present.

        edges:
            The row of every edge present in its edge group.
    """

    nodes: npt.NDArray[np.intp]
    sources: npt.NDArray[np.intp]
    targets: npt.NDArray[np.intp]
    edges: npt.NDArray[np.intp]

    def graph(self) -> nx.Graph:
        """Return the NetworkX graph of the snapshot, whose nodes are the node rows present.

        Returns:
            graph:
                The undirected graph of the nodes and edges present.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.specs import NetworkSnapshotSpec
            >>> rows = np.array([0, 1, 2])
            >>> snapshot = NetworkSnapshotSpec(rows, np.array([0, 1]), np.array([1, 2]), np.array([0, 1]))
            >>> sorted(snapshot.graph().edges())
            [(0, 1), (1, 2)]
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes.tolist())
        graph.add_edges_from(zip(self.sources.tolist(), self.targets.tolist(), strict=True))
        return graph
