"""Coarse-grain networks across scales by their Laplacian or by the angles of their nodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from ._networks import NetworkAnalysis
from .types import Source

KINDS = ('laplacian', 'geometric')
SETTINGS = {
    'laplacian': {'steps', 'time'},
    'geometric': {'steps', 'block', 'angle'},
}
"""Settings every kind reads, besides `kind`."""

MIN_BLOCK = 2
DENSE_NODES = 3000
SMALLEST_EIGENVALUE = 1e-10
TIMES = 200
SMALLEST_TIME_SCALE = 0.1
LARGEST_TIME_SCALE = 10.0
SMALLEST_PEAK = 1e-3


class RenormalizationAnalysis(BaseAnalysis):
    """Network at every scale of a coarse-graining of a graph, or of the last frame of a trajectory.

    The kinds merge nodes into supernodes, and two supernodes link when any of their nodes link:

    - `'laplacian'`: the diffusion operator `K = exp(-time L)` joins two nodes whose entry is at least the smaller of
      their diagonal entries, and every connected component of these joins is a supernode. It runs at `time`, or else at
      the smallest local peak of the specific heat `C = -dS / d log(time)`, of the entropy of the operator normalized by
      its trace, at which at least one pair of nodes joins, and a network with no such peak stays as it is.
    - `'geometric'`: every `block` consecutive nodes in the order of their `angle` form a supernode, at the mean angle
      of its nodes, and the last block holds the nodes left over.

    A trajectory gives the network of its last frame, with the values of the states and parameters of the group on its
    nodes as node attributes.

    Args:
        kind:
            The coarse-graining, one of the kinds above.

        steps:
            The number of times the network is coarse-grained.

        time:
            The diffusion time of `'laplacian'`, or `None` for the peak of the specific heat.

        block:
            The number of nodes of every supernode of `'geometric'`.

        angle:
            The node attribute that holds the angle of `'geometric'`.

    Attributes:
        graphs_:
            The network at every scale, the original first.

        degrees_:
            The degree of every node at every scale.

        exponents_:
            The fitted degree exponent at every scale, `nan` below 50 nodes or for a narrow distribution, as
            `NetworkAnalysis('degrees')` fits it.

        clustering_:
            The average clustering at every scale.

    Raises:
        AnalysisError:
            The kind is unknown, the kind does not read a changed setting, `steps` is below 1, `block` is below 2,
            `time` is not positive, raised by `check`, or a node has no angle, raised by `run`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.analysis import RenormalizationAnalysis
        >>> blocks = nx.disjoint_union(nx.complete_graph(20), nx.complete_graph(20))
        >>> blocks.add_edge(0, 20)
        >>> laplacian = RenormalizationAnalysis(steps=1).check().prepare(blocks).run()
        >>> sorted(laplacian.graphs_[1].degree())
        [(0, 1), (1, 1)]
        >>> ring = nx.cycle_graph(8)
        >>> nx.set_node_attributes(ring, {node: float(node) for node in ring}, 'angle')
        >>> geometric = RenormalizationAnalysis('geometric', steps=2).check().prepare(ring).run()
        >>> [graph.number_of_nodes() for graph in geometric.graphs_]
        [8, 4, 2]
    """

    def __init__(
        self,
        kind: str = 'laplacian',
        *,
        steps: int = 3,
        time: float | None = None,
        block: int = 2,
        angle: str = 'angle',
    ) -> None:
        self.kind = kind
        self.steps = steps
        self.time = time
        self.block = block
        self.angle = angle

    def _check(self) -> None:
        """Validate the kind, the steps, the time, the block, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        if self.steps < 1:
            error_msg = f'{type(self).__name__} has steps={self.steps!r}. Use 1 or more.'
            raise AnalysisError(error_msg)
        if self.block < MIN_BLOCK:
            error_msg = f'{type(self).__name__} has block={self.block!r}. Use {MIN_BLOCK} or more.'
            raise AnalysisError(error_msg)
        if self.time is not None and self.time <= 0:
            error_msg = f'{type(self).__name__} has time={self.time!r}. Use a positive time, or None.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the network kind."""
        return DataKind.NETWORK

    def _prepare(self, source: Source) -> None:
        """Read the graph, or the network of the last frame of a trajectory with its node states as attributes."""
        self.graph_ = self._read_frame(source) if isinstance(source, Trajectory) else self._read_networks(source)[-1]

    def _read_frame(self, trajectory: Trajectory) -> nx.Graph:
        """Return the network of the last frame of a trajectory with the values of the group on its nodes."""
        topology = trajectory.problem.topology
        if topology is None:
            return self._read_networks(trajectory)[-1]
        graph = trajectory.network(-1).graph()
        for name, (entry, _) in topology.networks[0].nodes[0].variables.items():
            values = np.asarray(trajectory.values(entry.name)[-1])
            nx.set_node_attributes(graph, {node: values[node] for node in graph}, name)
        return graph

    def _run(self, inputs: object = None) -> None:
        """Coarse-grain the network, and measure it at every scale."""
        self.graphs_ = [self.graph_.to_undirected()]
        for _ in range(self.steps):
            self.graphs_.append(self._coarsen(self.graphs_[-1]))
        measures = [NetworkAnalysis('degrees').check().prepare(graph).run() for graph in self.graphs_]
        self.degrees_ = [measure.degrees_[0] for measure in measures]
        self.exponents_ = np.asarray([measure.exponent_[0] for measure in measures], dtype=float)
        self.clustering_ = np.asarray([nx.average_clustering(graph) for graph in self.graphs_], dtype=float)

    def _coarsen(self, graph: nx.Graph) -> nx.Graph:
        """Return the network of the next scale."""
        if self.kind == 'geometric':
            return self._merge_by_angle(graph)
        nodes = list(graph)
        labels = self._diffusion_labels(graph, nodes)
        blocks: dict[int, set[object]] = {}
        for node, label in zip(nodes, labels.tolist(), strict=True):
            blocks.setdefault(label, set()).add(node)
        return nx.quotient_graph(graph, list(blocks.values()), relabel=True)

    def _merge_by_angle(self, graph: nx.Graph) -> nx.Graph:
        """Return the network of the blocks of consecutive nodes in the order of their angle, at their mean angle."""
        missing = [node for node in graph if self.angle not in graph.nodes[node]]
        if missing:
            error_msg = f'The nodes {missing[:5]} have no attribute {self.angle!r}. Give it, or set `angle`.'
            raise AnalysisError(error_msg)
        ordered = sorted(graph, key=lambda node: float(graph.nodes[node][self.angle]))
        blocks = [ordered[start : start + self.block] for start in range(0, len(ordered), self.block)]
        merged = nx.quotient_graph(graph, [set(block) for block in blocks], relabel=True)
        angles = [float(np.mean([graph.nodes[node][self.angle] for node in block])) for block in blocks]
        nx.set_node_attributes(merged, dict(enumerate(angles)), self.angle)
        return merged

    def _diffusion_labels(self, graph: nx.Graph, nodes: list[object]) -> npt.NDArray[np.int64]:
        """Return the supernode of every node, at the given time or at the first peak of the heat that merges nodes."""
        laplacian = nx.laplacian_matrix(graph, nodelist=nodes).astype(float).tocsc()
        if self.time is not None:
            return self._join_labels(self._diffuse(laplacian, self.time))
        singletons = np.arange(len(nodes), dtype=np.int64)
        for time in self._heat_peaks(np.linalg.eigvalsh(laplacian.toarray())):
            labels = self._join_labels(self._diffuse(laplacian, time))
            if labels.max() + 1 < len(nodes):
                return labels
        return singletons

    @staticmethod
    def _diffuse(laplacian: sparse.csc_matrix, time: float) -> Array:
        """Return the diffusion operator, with the dense exponential below a few thousand nodes."""
        size = laplacian.shape[0]
        if size < DENSE_NODES:
            return np.asarray(expm(-time * laplacian.toarray()), dtype=float)
        return np.asarray(expm_multiply(-time * laplacian, np.eye(size)), dtype=float)

    @staticmethod
    def _join_labels(operator: Array) -> npt.NDArray[np.int64]:
        """Return the connected component of every node, where an entry at least the smaller diagonal entry joins."""
        diagonal = np.diag(operator)
        joins = operator >= np.minimum.outer(diagonal, diagonal)
        return np.asarray(connected_components(sparse.csr_matrix(joins), directed=False)[1], dtype=np.int64)

    @staticmethod
    def _heat_peaks(eigenvalues: Array) -> list[float]:
        """Return the diffusion times of the local peaks of the specific heat, smallest first."""
        positive = eigenvalues[eigenvalues > SMALLEST_EIGENVALUE]
        if not positive.size:
            return []
        times = np.geomspace(SMALLEST_TIME_SCALE / positive.max(), LARGEST_TIME_SCALE / positive.min(), TIMES)
        weights = np.exp(-np.outer(times, eigenvalues))
        shares = weights / weights.sum(axis=1, keepdims=True)
        entropy = -(shares * np.log(np.where(shares > 0, shares, 1.0))).sum(axis=1)
        heat = -np.gradient(entropy, np.log(times))
        rising = (heat[1:-1] > heat[:-2]) & (heat[1:-1] >= heat[2:])
        tall = heat[1:-1] > SMALLEST_PEAK * heat.max()
        return times[1:-1][rising & tall].tolist()
