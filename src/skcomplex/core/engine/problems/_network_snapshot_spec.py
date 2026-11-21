"""Define the network snapshot spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt

from ...base import BaseSpec


@dataclass(frozen=True)
class NetworkSnapshotSpec(BaseSpec):
    """A spec of the nodes, links, and hyperedges of a network at one frame.

    It holds the nodes, the sources and targets of the links, the links themselves, and the hyperedges of a network
    whose links change, such as the 50 links added to 100 nodes by explosive percolation. `Trajectory.get_network`
    returns one, for a growing web, a changing friendship network, and a power grid that fails.

    Read more in the user guide, under Analyse.

    Args:
        nodes:
            The rows of the nodes present, in increasing order.

        sources:
            The node row of the source of every edge present.

        targets:
            The node row of the target of every edge present.

        edges:
            The row of every edge present in its edge group.

        hyperedges:
            The member nodes of every hyperedge present, one row per hyperedge, or `None` for a network whose
            hyperedges do not change.

    See Also:
        Trajectory: Returns it from `get_network`.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import ExplosivePercolationSystem
        >>> trajectory = ExplosivePercolationSystem(100).prepare().simulate(until=50, random_state=0)
        >>> len(trajectory.get_network(-1).sources)
        50
    """

    nodes: npt.NDArray[np.intp]
    sources: npt.NDArray[np.intp]
    targets: npt.NDArray[np.intp]
    edges: npt.NDArray[np.intp]
    hyperedges: npt.NDArray[np.intp] | None = None

    def get_graph(self) -> nx.Graph:
        """Return the NetworkX graph of the snapshot.

        Its nodes are the node rows present.

        Returns:
            graph:
                The undirected graph of the nodes and edges present.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.problems import NetworkSnapshotSpec
            >>> rows = np.array([0, 1, 2])
            >>> snapshot = NetworkSnapshotSpec(rows, np.array([0, 1]), np.array([1, 2]), np.array([0, 1]))
            >>> sorted(snapshot.get_graph().edges())
            [(0, 1), (1, 2)]
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes.tolist())
        graph.add_edges_from(zip(self.sources.tolist(), self.targets.tolist(), strict=True))
        return graph
