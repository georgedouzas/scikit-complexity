"""Measure clusters and percolation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from ..components.spaces import ContinuousSpace
from ..engine.runtime import Trajectory
from ..enums import DataKind
from ..errors import AnalysisError
from ..types import IntArray
from ._analysis_tags import AnalysisTags
from ._base import _check_choice, _check_settings, _get_missing_error, _read_fields
from .types import DataSource

KINDS = ('labels', 'sizes', 'percolation', 'spanning')
SETTINGS = {
    'labels': {'periodic', 'diagonal', 'axis', 'radius'},
    'sizes': {'periodic', 'diagonal', 'radius'},
    'percolation': {'diagonal', 'axis', 'radius'},
    'spanning': {'diagonal', 'axis', 'radius'},
}
TAGS = AnalysisTags(DataKind.FIELD)


@dataclass(frozen=True)
class ClustersResult:
    """A result of `clusters`, the clusters of a grid, a graph, or points.

    It holds the label of every cell, the size of every cluster, largest first, and for percolation the share of the
    largest cluster, the mean size, and whether a cluster joins two opposite edges, such as `[2, 1]` for a pair and a
    single. It reports the flow paths of a porous rock, the burnt patches of a forest, and the groups of like-minded
    voters.

    Read more in the user guide, under Analyse.

    Args:
        points:
            The arrays of points, one per entry of the first axis, or `None` for grids and graphs.

        faces:
            The sides of the space that holds the points of a trajectory, or `None` for the bounding box of the points.

        grids:
            The grids, one per entry of the first axis, or `None` for a graph or points.

        graph:
            The graph, or `None` for grids and points.

        labels:
            The label of every cell, 0 for empty cells and 1 onwards for clusters, for `'labels'` and for points, or
            the component of every node of a graph in node order, 1 for the largest, for `'sizes'` and
            `'percolation'`.

        sizes:
            The size of every cluster, in label order for `'labels'` and largest first otherwise, or `None` for
            `'spanning'`.

        members:
            The nodes of every component of a graph in node order, largest first, for `'sizes'` and `'percolation'`,
            or `None` for grids and points.

        strength:
            The share of all sites or nodes in the largest cluster, occupied or not, for `'percolation'` and for
            points. Dividing it by the occupation probability `p` gives the share of the occupied sites in the
            largest cluster, such as the `F` of Charbonneau.

        mean_size:
            The mean size of the cluster of an occupied site outside the largest, `Σ' s² / Σ' s`, for `'percolation'`
            and for points.

        spanning:
            Whether a cluster of the grid joins the two opposite edges of `axis`, always false for graphs, for
            `'labels'`, `'percolation'`, and points.

        probability:
            The spanning probability of the ensemble, for `'spanning'`.

    See Also:
        clusters: Returns this result.

    Examples:
        >>> from skcomplex.core.analysis import clusters
        >>> clusters([[1, 1, 0], [0, 0, 1]], kind='sizes').sizes.tolist()
        [2, 1]
        >>> import networkx as nx
        >>> clusters(nx.Graph([(0, 1), (2, 3), (3, 4)]), kind='sizes').members
        [[2, 3, 4], [0, 1]]
    """

    points: npt.NDArray[np.float64] | None = None
    faces: npt.NDArray[np.float64] | None = None
    grids: npt.NDArray[np.bool_] | None = None
    graph: nx.Graph | None = None
    labels: IntArray | None = None
    sizes: IntArray | None = None
    members: list[list[object]] | None = None
    strength: float | None = None
    mean_size: float | None = None
    spanning: bool | None = None
    probability: float | None = None


def clusters(
    source: DataSource,
    *,
    kind: str = 'labels',
    variables: Sequence[str] | None = None,
    periodic: bool = False,
    diagonal: bool = False,
    axis: int = 0,
    radius: float | None = None,
) -> ClustersResult:
    """Return the clusters of a grid, a graph, or points.

    A cluster is a set of occupied cells of a grid that touch through faces, or corners with `diagonal=True`, the
    connected nodes of a graph, or points closer than twice `radius`, such as the clusters of sizes 2 and 1 in
    `[[1, 1, 0], [0, 0, 1]]`. A trajectory gives its last field, network, or positions. A cluster spans when it joins
    two opposite edges along `axis`. It finds the path of water through a porous rock, the burnt patches of a forest
    fire, and the groups of an opinion model.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The grid or the grids, the graph, the points, or the trajectory.

        kind:
            What to return. `'labels'` gives the label of every cell, the size of every cluster in label order, and
            whether a cluster spans. `'sizes'` gives the sizes, largest first, and for a graph the component of
            every node and the nodes of every component. `'percolation'` gives the sizes, the
            strength, the share of all sites in the largest cluster, the mean size, and spanning. The strength divided
            by the occupation probability `p` is the share of the occupied sites in the largest cluster. `'spanning'`
            gives the share of the grids of an ensemble, stacked on the first axis, with a spanning cluster.

        variables:
            The field of a trajectory to read, or `None` for its first field.

        periodic:
            Whether clusters of a grid wrap around its edges.

        diagonal:
            Whether cells that share only a corner are connected.

        axis:
            The axis whose two opposite edges a spanning cluster joins.

        radius:
            The contact radius of points, or `None` for grids and graphs.

    Returns:
        result:
            What the source was read as, and the measures of the kind.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, the radius is not positive or comes with
            settings of grids, or a kind that reads grids gets a graph.

    See Also:
        domains: Finds the regions of equal value of a field.
        defects: Finds the defects of a field of angles.

    Examples:
        >>> from skcomplex.core.analysis import clusters
        >>> result = clusters([[1, 1, 0], [1, 0, 1]], kind='percolation')
        >>> result.strength, result.mean_size, result.spanning
        (0.5, 1.0, True)
    """
    _check_choice('clusters', 'kind', kind, KINDS)
    _check_settings(
        clusters,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        periodic=periodic,
        diagonal=diagonal,
        axis=axis,
        radius=radius,
    )
    if radius is not None:
        if radius <= 0 or periodic or diagonal:
            error_msg = (
                f'clusters has radius={radius!r}, periodic={periodic!r}, and diagonal={diagonal!r}. A radius is '
                'positive, and points take neither of the settings of grids.'
            )
            raise AnalysisError(error_msg)
        points, faces = _read_points(source, kind, variables)
        return _of_points(points, faces, kind, axis, radius)
    if isinstance(source, nx.Graph):
        if kind in {'labels', 'spanning'}:
            error_msg = f'clusters of kind {kind!r} reads grids, not graphs. Use the kind sizes or percolation.'
            raise AnalysisError(error_msg)
        return _of_graph(source, kind)
    name = variables[0] if variables else None
    read = np.asarray(_read_fields(source, name), dtype=bool)
    if kind == 'spanning':
        spans = [_spans(_label(grid, periodic=False, diagonal=diagonal)[0], axis) for grid in read]
        return ClustersResult(grids=read, probability=float(np.mean(spans)))
    grids = read[-1:] if isinstance(source, Trajectory) else read[None]
    if kind == 'labels':
        labels, sizes = _label(grids[0], periodic=periodic, diagonal=diagonal)
        return ClustersResult(grids=grids, labels=labels, sizes=sizes, spanning=_spans(labels, axis))
    if kind == 'sizes':
        _, sizes = _label(grids[0], periodic=periodic, diagonal=diagonal)
        return ClustersResult(grids=grids, sizes=np.sort(sizes)[::-1])
    labels, sizes = _label(grids[0], periodic=False, diagonal=diagonal)
    ordered = np.sort(sizes)[::-1]
    strength, mean_size = _strength(ordered, grids[0].size)
    return ClustersResult(
        grids=grids,
        sizes=ordered,
        strength=strength,
        mean_size=mean_size,
        spanning=_spans(labels, axis),
    )


def _of_graph(graph: nx.Graph, kind: str) -> ClustersResult:
    """Return the sizes, the labels, and the members of the components of a graph.

    The components come largest first, ties by their first node in node order, and for percolation the strength and
    the mean size are returned too.
    """
    position = {node: index for index, node in enumerate(graph.nodes())}
    members = sorted(
        (sorted(component, key=position.__getitem__) for component in nx.connected_components(graph.to_undirected())),
        key=lambda component: (-len(component), position[component[0]]),
    )
    sizes = np.asarray([len(component) for component in members], dtype=np.int64)
    label = {node: index + 1 for index, component in enumerate(members) for node in component}
    labels = np.asarray([label[node] for node in graph.nodes()], dtype=np.int64)
    if kind == 'sizes':
        return ClustersResult(graph=graph, labels=labels, sizes=sizes, members=members)
    strength, mean_size = _strength(sizes, graph.number_of_nodes())
    return ClustersResult(
        graph=graph,
        labels=labels,
        sizes=sizes,
        members=members,
        strength=strength,
        mean_size=mean_size,
        spanning=False,
    )


def _strength(sizes: IntArray, total: int) -> tuple[float, float]:
    """Return the share in a total of the largest of sizes.

    The sizes come largest first, and the mean size of the others is returned too.
    """
    rest = sizes[1:].astype(float)
    strength = float(sizes[0] / total) if sizes.size else 0.0
    return strength, float(np.sum(rest**2) / rest.sum()) if rest.size else 0.0


def _read_points(
    source: DataSource,
    kind: str,
    variables: Sequence[str] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Return the arrays of points of the source and the faces of the space.

    The space is the one that holds them.
    """
    if not isinstance(source, Trajectory):
        points = np.asarray(source, dtype=float)
        return (points if kind == 'spanning' else points[None]), None
    system = source.problem.system
    space = None if system is None else system.space
    names = [name for name in source.problem.states.names if source.problem.states.get_entry(name).columns > 1]
    name = variables[0] if variables else (names[0] if names else None)
    if name is None:
        raise _get_missing_error(source, 'vector state of positions')
    faces = np.asarray(space.extent, dtype=float) if isinstance(space, ContinuousSpace) and space.extent else None
    return np.asarray(source.get_values(name), dtype=float)[-1:], faces


def _contacts(points: npt.NDArray[np.float64], radius: float) -> IntArray:
    """Return the label of every point in the graph of close points.

    Labels start from 0, and two points are close when they are closer than twice the radius.
    """
    pairs = cKDTree(points).query_pairs(2 * radius, output_type='ndarray')
    size = len(points)
    graph = sparse.coo_array((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(size, size))
    return csgraph.connected_components(graph, directed=False)[1]


def _spans_points(
    points: npt.NDArray[np.float64],
    labels: IntArray,
    faces: npt.NDArray[np.float64] | None,
    axis: int,
    radius: float,
) -> bool:
    """Return whether one cluster holds points within the radius of both faces.

    The faces are those along the axis.
    """
    along = points[:, axis]
    lower, upper = (0.0, float(faces[axis])) if faces is not None else (along.min(), along.max())
    return bool(set(labels[along <= lower + radius]) & set(labels[along >= upper - radius]))


def _of_points(
    ensemble: npt.NDArray[np.float64],
    faces: npt.NDArray[np.float64] | None,
    kind: str,
    axis: int,
    radius: float,
) -> ClustersResult:
    """Return the measure of the kind from the contact clusters of points."""
    if kind == 'spanning':
        spans = [_spans_points(item, _contacts(item, radius), faces, axis, radius) for item in ensemble]
        return ClustersResult(points=ensemble, faces=faces, probability=float(np.mean(spans)))
    points = ensemble[0]
    labels = _contacts(points, radius)
    counts = np.bincount(labels)
    sizes = counts if kind == 'labels' else np.sort(counts)[::-1]
    rest = sizes[1:].astype(float)
    return ClustersResult(
        points=ensemble,
        faces=faces,
        labels=labels + 1,
        sizes=sizes,
        strength=float(counts.max() / len(points)),
        mean_size=float(np.sum(rest**2) / rest.sum()) if rest.size else 0.0,
        spanning=_spans_points(points, labels, faces, axis, radius),
    )


def _merge(labels: IntArray, structure: npt.NDArray[np.bool_]) -> IntArray:
    """Return the labels with the clusters joined across the periodic edges.

    They are joined by a neighbour of the structure.
    """
    centre = np.ones(structure.ndim, dtype=int)
    sources, targets = [], []
    for offset in np.argwhere(structure) - centre:
        shifted = np.roll(labels, tuple(offset), axis=tuple(range(labels.ndim)))
        both = (labels > 0) & (shifted > 0)
        sources.append(labels[both])
        targets.append(shifted[both])
    nodes = int(labels.max()) + 1
    rows, columns = np.concatenate(sources), np.concatenate(targets)
    adjacency = sparse.coo_array((np.ones(rows.size), (rows, columns)), shape=(nodes, nodes))
    _, components = csgraph.connected_components(adjacency, directed=False)
    return np.where(labels > 0, components[labels] + 1, 0)


def _label(grid: npt.ArrayLike, *, periodic: bool, diagonal: bool) -> tuple[IntArray, IntArray]:
    """Return the label of each cell and the size of each cluster.

    The label is 0 for empty cells and 1 onwards for clusters.
    """
    occupied = np.asarray(grid, dtype=bool)
    structure = ndimage.generate_binary_structure(occupied.ndim, occupied.ndim if diagonal else 1)
    labels, _ = ndimage.label(occupied, structure=structure)
    if periodic:
        labels = _merge(labels, structure)
    distinct = np.unique(labels[labels > 0])
    relabelled = np.zeros_like(labels)
    relabelled[labels > 0] = np.searchsorted(distinct, labels[labels > 0]) + 1
    return relabelled, np.bincount(relabelled.ravel())[1:]


def _spans(labels: IntArray, axis: int) -> bool:
    """Return whether one cluster touches both opposite edges along an axis."""
    first = set(np.take(labels, 0, axis=axis).ravel()) - {0}
    last = set(np.take(labels, -1, axis=axis).ravel()) - {0}
    return bool(first & last)
