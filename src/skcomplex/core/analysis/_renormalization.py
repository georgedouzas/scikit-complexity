"""Coarse-grain networks across scales.

The coarse-graining is by their Laplacian or by the angles of their nodes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _check_tags, _read_networks
from ._networks import network_measures
from .types import Source

KINDS = ('laplacian', 'geometric')
SETTINGS = {
    'laplacian': {'steps', 'time'},
    'geometric': {'steps', 'block', 'angle', 'beta'},
}
"""Settings every kind reads, besides `kind`."""

MIN_BLOCK = 2
DENSE_NODES = 3000
SMALLEST_EIGENVALUE = 1e-10
TIMES = 200
SMALLEST_TIME_SCALE = 0.1
LARGEST_TIME_SCALE = 10.0
SMALLEST_PEAK = 1e-3

TAGS = AnalysisTags(DataKind.NETWORK, space=frozenset({'network'}))


@dataclass(frozen=True)
class RenormalizationResult:
    """Network at every scale of a coarse-graining of a graph or of a trajectory.

    The graph of a trajectory is that of its last frame.

    Args:
        graph:
            The network read from the source.

        graphs:
            The network at every scale, the original first.

        degrees:
            The degree of every node at every scale.

        exponents:
            The fitted degree exponent at every scale, `nan` below 50 nodes or for a narrow distribution, as
            `network_measures(kind='degrees')` fits it.

        clustering:
            The average clustering at every scale.
    """

    graph: nx.Graph
    graphs: list[nx.Graph]
    degrees: list[npt.NDArray[np.int64]]
    exponents: Array
    clustering: Array


def renormalization(
    source: Source,
    *,
    kind: str = 'laplacian',
    steps: int = 3,
    time: float | None = None,
    block: int = 2,
    angle: str = 'angle',
    beta: float = 1.0,
) -> RenormalizationResult:
    """Return the network at every scale of a coarse-graining of a graph or trajectory.

    The graph of a trajectory is that of its last frame.

    The kinds merge nodes into supernodes, and two supernodes link when any of their nodes link:

    - `'laplacian'`: the diffusion operator `K = exp(-time L)` joins two nodes whose entry is at least the smaller of
      their diagonal entries, and every connected component of these joins is a supernode. It runs at `time`, or else at
      the smallest local peak of the specific heat `C = -dS / d log(time)`, of the entropy of the operator normalized by
      its trace, at which at least one pair of nodes joins, and a network with no such peak stays as it is.
    - `'geometric'`: every `block` consecutive nodes in the order of their `angle` form a supernode, and the last block
      holds the nodes left over. The supernode sits at the mean angle of its nodes weighted by `z = k^max(1, beta)`,
      with the degree `k` of a node, which is proportional to its hidden degree in the geometric system.

    A trajectory gives the network of its last frame, with the values of the states and parameters of the group on its
    nodes as node attributes.

    Args:
        source:
            The trajectory, a graph, or a network snapshot.

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

        beta:
            The parameter of the geometric system that the weights of the angles of `'geometric'` read, the inverse of
            its temperature.

    Returns:
        result:
            The network at every scale and its measures.

    Raises:
        AnalysisError:
            The kind is unknown, the kind does not read a changed setting, `steps` is below 1, `block` is below 2,
            `time` or `beta` is not positive, or a node has no angle.

        TagError:
            The system of a trajectory has no network space.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import renormalization
        >>> blocks = nx.disjoint_union(nx.complete_graph(20), nx.complete_graph(20))
        >>> blocks.add_edge(0, 20)
        >>> sorted(renormalization(blocks, steps=1).graphs[1].degree())
        [(0, 1), (1, 1)]
        >>> ring = nx.cycle_graph(8)
        >>> nx.set_node_attributes(ring, {node: float(node) for node in ring}, 'angle')
        >>> [graph.number_of_nodes() for graph in renormalization(ring, kind='geometric', steps=2).graphs]
        [8, 4, 2]
    """
    _check_choice('renormalization', 'kind', kind, KINDS)
    _check_tags(renormalization, source, kind)
    _check_settings(
        renormalization,
        'kind',
        SETTINGS[kind],
        kind=kind,
        steps=steps,
        time=time,
        block=block,
        angle=angle,
        beta=beta,
    )
    if steps < 1:
        error_msg = f'renormalization has steps={steps!r}. Use 1 or more.'
        raise AnalysisError(error_msg)
    if block < MIN_BLOCK:
        error_msg = f'renormalization has block={block!r}. Use {MIN_BLOCK} or more.'
        raise AnalysisError(error_msg)
    if time is not None and time <= 0:
        error_msg = f'renormalization has time={time!r}. Use a positive time, or None.'
        raise AnalysisError(error_msg)
    if beta <= 0:
        error_msg = f'renormalization has beta={beta!r}. Use a positive value.'
        raise AnalysisError(error_msg)
    graph = _read_frame(source) if isinstance(source, Trajectory) else _read_networks(source)[-1]
    graphs = [graph.to_undirected()]
    for _ in range(steps):
        if kind == 'geometric':
            graphs.append(_merge_by_angle(graphs[-1], angle, block, beta))
        else:
            graphs.append(_coarsen(graphs[-1], time))
    measures = [network_measures(item, kind='degrees') for item in graphs]
    return RenormalizationResult(
        graph,
        graphs,
        degrees=[measure.degrees[0] for measure in measures if measure.degrees is not None],
        exponents=np.asarray(
            [measure.exponent[0] for measure in measures if measure.exponent is not None],
            dtype=float,
        ),
        clustering=np.asarray([nx.average_clustering(item) for item in graphs], dtype=float),
    )


def _read_frame(trajectory: Trajectory) -> nx.Graph:
    """Return the network of the last frame of a trajectory.

    The values of the group lie on its nodes.
    """
    topology = trajectory.problem.topology
    if topology is None:
        return _read_networks(trajectory)[-1]
    graph = trajectory.network(-1).graph()
    for name, (entry, _) in topology.networks[0].nodes[0].variables.items():
        values = np.asarray(trajectory.values(entry.name)[-1])
        nx.set_node_attributes(graph, {node: values[node] for node in graph}, name)
    return graph


def _coarsen(graph: nx.Graph, time: float | None) -> nx.Graph:
    """Return the network of the next scale of the Laplacian coarse-graining."""
    nodes = list(graph)
    labels = _diffusion_labels(graph, nodes, time)
    blocks: dict[int, set[object]] = {}
    for node, label in zip(nodes, labels.tolist(), strict=True):
        blocks.setdefault(label, set()).add(node)
    return nx.quotient_graph(graph, list(blocks.values()), relabel=True)


def _merge_by_angle(graph: nx.Graph, angle: str, block: int, beta: float) -> nx.Graph:
    """Return the network of the blocks of consecutive nodes by angle.

    The blocks lie at their angles weighted by the degrees.
    """
    missing = [node for node in graph if angle not in graph.nodes[node]]
    if missing:
        error_msg = f'The nodes {missing[:5]} have no attribute {angle!r}. Give it, or set `angle`.'
        raise AnalysisError(error_msg)
    ordered = sorted(graph, key=lambda node: float(graph.nodes[node][angle]))
    blocks = [ordered[start : start + block] for start in range(0, len(ordered), block)]
    merged = nx.quotient_graph(graph, [set(members) for members in blocks], relabel=True)
    angles = [_supernode_angle(graph, members, angle, beta) for members in blocks]
    nx.set_node_attributes(merged, dict(enumerate(angles)), angle)
    return merged


def _supernode_angle(graph: nx.Graph, block: list[object], angle: str, beta: float) -> float:
    """Return the angle of a supernode.

    It is the plain mean when none of its nodes has an edge.
    """
    angles = np.asarray([graph.nodes[node][angle] for node in block], dtype=float)
    degrees = np.asarray([graph.degree(node) for node in block], dtype=float)
    weights = degrees ** max(1.0, beta)
    return float(np.average(angles, weights=weights if weights.sum() > 0 else None))


def _diffusion_labels(graph: nx.Graph, nodes: list[object], time: float | None) -> npt.NDArray[np.int64]:
    """Return the supernode of every node.

    The nodes merge at the given time or at the first peak of the heat.
    """
    laplacian = nx.laplacian_matrix(graph, nodelist=nodes).astype(float).tocsc()
    if time is not None:
        return _join_labels(_diffuse(laplacian, time))
    singletons = np.arange(len(nodes), dtype=np.int64)
    for peak in _heat_peaks(np.linalg.eigvalsh(laplacian.toarray())):
        labels = _join_labels(_diffuse(laplacian, peak))
        if labels.max() + 1 < len(nodes):
            return labels
    return singletons


def _diffuse(laplacian: sparse.csc_matrix, time: float) -> Array:
    """Return the diffusion operator.

    It uses the dense exponential below a few thousand nodes.
    """
    size = laplacian.shape[0]
    if size < DENSE_NODES:
        return np.asarray(expm(-time * laplacian.toarray()), dtype=float)
    return np.asarray(expm_multiply(-time * laplacian, np.eye(size)), dtype=float)


def _join_labels(operator: Array) -> npt.NDArray[np.int64]:
    """Return the connected component of every node.

    An entry at least the smaller diagonal entry joins two nodes.
    """
    diagonal = np.diag(operator)
    joins = operator >= np.minimum.outer(diagonal, diagonal)
    return np.asarray(connected_components(sparse.csr_matrix(joins), directed=False)[1], dtype=np.int64)


def _heat_peaks(eigenvalues: Array) -> list[float]:
    """Return the diffusion times of the local peaks of the specific heat.

    The smallest comes first.
    """
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
