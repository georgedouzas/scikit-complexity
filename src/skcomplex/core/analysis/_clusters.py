"""Measure clusters and percolation: clusters of grids, graphs, and points, the largest cluster, and spanning."""

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
from ..errors import AnalysisError
from ..types import IntArray
from ._base import _check_choice, _check_settings, _get_missing_error, _read_fields
from .types import Source

KINDS = ('labels', 'sizes', 'percolation', 'spanning')
# Settings every kind reads, besides `kind` and `variables`.
SETTINGS = {
    'labels': {'periodic', 'diagonal', 'axis', 'radius'},
    'sizes': {'periodic', 'diagonal', 'radius'},
    'percolation': {'diagonal', 'axis', 'radius'},
    'spanning': {'diagonal', 'axis', 'radius'},
}


@dataclass(frozen=True)
class ClustersResult:
    """Clusters of the occupied cells of a grid, the components of a graph, or the contacts of points.

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
            The label of every cell, 0 for empty cells and 1 onwards for clusters, for `'labels'` and for points.

        sizes:
            The size of every cluster, in label order for `'labels'` and largest first otherwise, or `None` for
            `'spanning'`.

        strength:
            The share of all sites or nodes in the largest cluster, for `'percolation'` and for points.

        mean_size:
            The mean size of the cluster of an occupied site outside the largest, `Σ' s² / Σ' s`, for `'percolation'`
            and for points.

        spanning:
            Whether a cluster of the grid joins the two opposite edges of `axis`, always false for graphs, for
            `'labels'`, `'percolation'`, and points.

        probability:
            The spanning probability of the ensemble, for `'spanning'`.
    """

    points: npt.NDArray[np.float64] | None = None
    faces: npt.NDArray[np.float64] | None = None
    grids: npt.NDArray[np.bool_] | None = None
    graph: nx.Graph | None = None
    labels: IntArray | None = None
    sizes: IntArray | None = None
    strength: float | None = None
    mean_size: float | None = None
    spanning: bool | None = None
    probability: float | None = None


def clusters(
    source: Source,
    *,
    kind: str = 'labels',
    variables: Sequence[str] | None = None,
    periodic: bool = False,
    diagonal: bool = False,
    axis: int = 0,
    radius: float | None = None,
) -> ClustersResult:
    """Return the clusters of the occupied cells of a grid, the components of a graph, or the contacts of points.

    The grid is a boolean array of any dimension, the last frame of a field of a trajectory, or a graph, the last
    network of a trajectory. Cells are connected through shared faces, or also corners with `diagonal=True`.

    With a `radius`, the source holds points instead, an array with one row per point, a stack of such arrays for the
    kind `'spanning'`, or the positions of a group of a trajectory in its last frame. Points closer than twice the
    radius are connected, and a cluster joins two opposite faces along `axis` when it holds points within the radius of
    both. The faces are those of the space of a trajectory, and those of the bounding box of an array.

    - `'labels'`: the label of every cell and the size of every cluster of a grid, in label order, and whether a
      cluster joins the two opposite edges of `axis`.
    - `'sizes'`: the size of every cluster of a grid or of every component of a graph, largest first.
    - `'percolation'`: the cluster sizes, the strength of the largest cluster, the mean cluster size, and spanning.
    - `'spanning'`: the share of the grids of an ensemble, stacked on the first axis or the frames of a trajectory,
      with a cluster joining the opposite edges of `axis`.

    Args:
        source:
            The grid or the grids, the graph, the points, or the trajectory.

        kind:
            The measure to compute, one of the kinds above.

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

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.analysis import clusters
        >>> clusters([[1, 1, 0], [0, 0, 1]], kind='sizes').sizes.tolist()
        [2, 1]
        >>> clusters(nx.path_graph(3), kind='sizes').sizes.tolist()
        [3]
        >>> result = clusters([[1, 1, 0], [1, 0, 1]], kind='percolation')
        >>> result.strength, result.mean_size, result.spanning
        (0.5, 1.0, True)
        >>> clusters([[0, 0], [1, 0], [5, 5]], kind='sizes', radius=0.6).sizes.tolist()
        [2, 1]
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
    """Return the sizes of the components of a graph, and for percolation the strength and the mean size."""
    components = nx.connected_components(graph.to_undirected(as_view=True))
    sizes = np.sort(np.asarray([len(component) for component in components], dtype=np.int64))[::-1]
    if kind == 'sizes':
        return ClustersResult(graph=graph, sizes=sizes)
    strength, mean_size = _strength(sizes, graph.number_of_nodes())
    return ClustersResult(graph=graph, sizes=sizes, strength=strength, mean_size=mean_size, spanning=False)


def _strength(sizes: IntArray, total: int) -> tuple[float, float]:
    """Return the share in a total of the largest of sizes, largest first, and the mean size of the others."""
    rest = sizes[1:].astype(float)
    strength = float(sizes[0] / total) if sizes.size else 0.0
    return strength, float(np.sum(rest**2) / rest.sum()) if rest.size else 0.0


def _read_points(
    source: Source,
    kind: str,
    variables: Sequence[str] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Return the arrays of points of the source and the faces of the space that holds them."""
    if not isinstance(source, Trajectory):
        points = np.asarray(source, dtype=float)
        return (points if kind == 'spanning' else points[None]), None
    system = source.problem.system
    space = None if system is None else system.space
    names = [name for name in source.problem.states.names if source.problem.states.entry(name).columns > 1]
    name = variables[0] if variables else (names[0] if names else None)
    if name is None:
        raise _get_missing_error(source, 'vector state of positions')
    faces = np.asarray(space.extent, dtype=float) if isinstance(space, ContinuousSpace) and space.extent else None
    return np.asarray(source.values(name), dtype=float)[-1:], faces


def _contacts(points: npt.NDArray[np.float64], radius: float) -> IntArray:
    """Return the label of every point, from 0, in the graph of the points closer than twice the radius."""
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
    """Return whether one cluster holds points within the radius of both faces along the axis."""
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
    """Return the labels with the clusters joined by a neighbour of the structure across the periodic edges."""
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
    """Return the label of each cell, 0 for empty cells and 1 onwards for clusters, and the size of each cluster."""
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
