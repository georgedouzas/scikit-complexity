"""Measure clusters and percolation: clusters of grids, graphs, and points, the largest cluster, and spanning."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from ..core.spaces import ContinuousSpace
from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import DataKind, IntArray
from ._base import BaseAnalysis
from .types import Source

KINDS = ('labels', 'sizes', 'percolation', 'spanning')
SETTINGS = {
    'labels': {'periodic', 'diagonal', 'axis', 'radius'},
    'sizes': {'periodic', 'diagonal', 'radius'},
    'percolation': {'diagonal', 'axis', 'radius'},
    'spanning': {'diagonal', 'axis', 'radius'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


class ClusterAnalysis(BaseAnalysis):
    """Clusters of the occupied cells of a grid, the components of a graph, or the contacts of points, of one kind.

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

    Attributes:
        points_:
            The arrays of points, one per entry of the first axis, or `None` for grids and graphs.

        grids_:
            The grids, one per entry of the first axis, or `None` for a graph.

        graph_:
            The graph, or `None` for grids.

        labels_:
            The label of every cell, 0 for empty cells and 1 onwards for clusters.

        sizes_:
            The size of every cluster, in label order for `'labels'` and largest first otherwise.

        strength_:
            The share of all sites or nodes in the largest cluster.

        mean_size_:
            The mean size of the cluster of an occupied site outside the largest, `Σ' s² / Σ' s`.

        spanning_:
            Whether a cluster of the grid joins the two opposite edges of `axis`, always false for graphs.

        probability_:
            The spanning probability of the ensemble.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.analysis import ClusterAnalysis
        >>> ClusterAnalysis('sizes').check().prepare([[1, 1, 0], [0, 0, 1]]).run().sizes_.tolist()
        [2, 1]
        >>> ClusterAnalysis('sizes').check().prepare(nx.path_graph(3)).run().sizes_.tolist()
        [3]
        >>> result = ClusterAnalysis('percolation').check().prepare([[1, 1, 0], [1, 0, 1]]).run()
        >>> result.strength_, result.mean_size_, result.spanning_
        (0.5, 1.0, True)
        >>> ClusterAnalysis('sizes', radius=0.6).check().prepare([[0, 0], [1, 0], [5, 5]]).run().sizes_.tolist()
        [2, 1]
    """

    def __init__(
        self,
        kind: str = 'labels',
        variables: Sequence[str] | None = None,
        *,
        periodic: bool = False,
        diagonal: bool = False,
        axis: int = 0,
        radius: float | None = None,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.periodic = periodic
        self.diagonal = diagonal
        self.axis = axis
        self.radius = radius

    def _check(self) -> None:
        """Validate the kind, the radius, and that the kind reads every changed setting."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.radius is None:
            return
        if self.radius <= 0 or self.periodic or self.diagonal:
            error_msg = (
                f'{type(self).__name__} has radius={self.radius!r}, periodic={self.periodic!r}, and '
                f'diagonal={self.diagonal!r}. A radius is positive, and points take neither of the settings of grids.'
            )
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the field kind."""
        return DataKind.FIELD

    def _prepare(self, source: Source) -> None:
        """Read a graph, or the grids of the source, the last frame of a trajectory but for spanning."""
        self.grids_: npt.NDArray[np.bool_] | None = None
        self.graph_: nx.Graph | None = None
        self.points_: npt.NDArray[np.float64] | None = None
        if self.radius is not None:
            self._prepare_points(source)
            return
        if isinstance(source, nx.Graph):
            if self.kind in {'labels', 'spanning'}:
                error_msg = (
                    f'{type(self).__name__} of kind {self.kind!r} reads grids, not graphs. Use the kind sizes or '
                    'percolation.'
                )
                raise AnalysisError(error_msg)
            self.graph_ = source
            return
        name = self.variables[0] if self.variables else None
        grids = np.asarray(self._read_fields(source, name), dtype=bool)
        if self.kind == 'spanning':
            self.grids_ = grids
        else:
            self.grids_ = grids[-1:] if isinstance(source, Trajectory) else grids[None]

    def _prepare_points(self, source: Source) -> None:
        """Read the arrays of points of the source and the faces of the space that holds them."""
        self.faces_: npt.NDArray[np.float64] | None = None
        if not isinstance(source, Trajectory):
            points = np.asarray(source, dtype=float)
            self.points_ = points if self.kind == 'spanning' else points[None]
            return
        system = source.problem.system
        space = None if system is None else system.space
        names = [name for name in source.problem.states.names if source.problem.states.entry(name).columns > 1]
        name = self.variables[0] if self.variables else (names[0] if names else None)
        if name is None:
            raise self._get_missing_error(source, 'vector state of positions')
        self.points_ = np.asarray(source.values(name), dtype=float)[-1:]
        if isinstance(space, ContinuousSpace) and space.extent:
            self.faces_ = np.asarray(space.extent, dtype=float)

    def _contacts(self, points: npt.NDArray[np.float64]) -> IntArray:
        """Return the label of every point, from 0, in the graph of the points closer than twice the radius."""
        pairs = cKDTree(points).query_pairs(2 * float(self.radius or 0.0), output_type='ndarray')
        size = len(points)
        graph = sparse.coo_array((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(size, size))
        return csgraph.connected_components(graph, directed=False)[1]

    def _spans_points(self, points: npt.NDArray[np.float64], labels: IntArray) -> bool:
        """Return whether one cluster holds points within the radius of both faces along the axis."""
        along = points[:, self.axis]
        lower, upper = (0.0, float(self.faces_[self.axis])) if self.faces_ is not None else (along.min(), along.max())
        radius = float(self.radius or 0.0)
        return bool(set(labels[along <= lower + radius]) & set(labels[along >= upper - radius]))

    def _run_points(self, ensemble: npt.NDArray[np.float64]) -> None:
        """Compute the measure of the kind from the contact clusters of points."""
        if self.kind == 'spanning':
            self.probability_ = float(np.mean([self._spans_points(item, self._contacts(item)) for item in ensemble]))
            return
        points = ensemble[0]
        labels = self._contacts(points)
        sizes = np.bincount(labels)
        self.spanning_ = self._spans_points(points, labels)
        self.labels_, self.sizes_ = labels + 1, sizes if self.kind == 'labels' else np.sort(sizes)[::-1]
        rest = self.sizes_[1:].astype(float)
        self.strength_ = float(sizes.max() / len(points))
        self.mean_size_ = float(np.sum(rest**2) / rest.sum()) if rest.size else 0.0

    def _get_grids(self) -> npt.NDArray[np.bool_]:
        """Return the grids the analysis read, raising an error for a graph, which has none."""
        if self.grids_ is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} reads grids, but the source is a graph.'
            raise AnalysisError(error_msg)
        return self.grids_

    def _run_percolation(self) -> None:
        """Compute the cluster sizes, the strength of the largest cluster, the mean cluster size, and spanning."""
        if self.graph_ is not None:
            self._run_sizes()
            total, self.spanning_ = self.graph_.number_of_nodes(), False
        else:
            labels, sizes = self._label(self._get_grids()[0], periodic=False, diagonal=self.diagonal)
            self.sizes_ = np.sort(sizes)[::-1]
            total, self.spanning_ = self._get_grids()[0].size, self._spans(labels, self.axis)
        rest = self.sizes_[1:].astype(float)
        self.strength_ = float(self.sizes_[0] / total) if self.sizes_.size else 0.0
        self.mean_size_ = float(np.sum(rest**2) / rest.sum()) if rest.size else 0.0

    def _run_sizes(self) -> None:
        """Compute the size of every cluster of the grid or of every component of the graph, largest first."""
        if self.graph_ is not None:
            components = nx.connected_components(self.graph_.to_undirected(as_view=True))
            sizes = np.asarray([len(component) for component in components], dtype=np.int64)
        else:
            _, sizes = self._label(self._get_grids()[0], periodic=self.periodic, diagonal=self.diagonal)
        self.sizes_ = np.sort(sizes)[::-1]

    def _run(self, inputs: object = None) -> None:
        """Compute the clusters, their sizes, the percolation observables, or the spanning probability."""
        if self.points_ is not None:
            self._run_points(self.points_)
        elif self.kind == 'labels':
            grid = self._get_grids()[0]
            self.labels_, self.sizes_ = self._label(grid, periodic=self.periodic, diagonal=self.diagonal)
            self.spanning_ = self._spans(self.labels_, self.axis)
        elif self.kind == 'sizes':
            self._run_sizes()
        elif self.kind == 'percolation':
            self._run_percolation()
        else:
            spans = [
                self._spans(self._label(grid, periodic=False, diagonal=self.diagonal)[0], self.axis)
                for grid in self._get_grids()
            ]
            self.probability_ = float(np.mean(spans))

    @staticmethod
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

    @staticmethod
    def _label(grid: npt.ArrayLike, *, periodic: bool, diagonal: bool) -> tuple[IntArray, IntArray]:
        """Return the label of each cell, 0 for empty cells and 1 onwards for clusters, and the size of each cluster."""
        occupied = np.asarray(grid, dtype=bool)
        structure = ndimage.generate_binary_structure(occupied.ndim, occupied.ndim if diagonal else 1)
        labels, _ = ndimage.label(occupied, structure=structure)
        if periodic:
            labels = ClusterAnalysis._merge(labels, structure)
        distinct = np.unique(labels[labels > 0])
        relabelled = np.zeros_like(labels)
        relabelled[labels > 0] = np.searchsorted(distinct, labels[labels > 0]) + 1
        return relabelled, np.bincount(relabelled.ravel())[1:]

    @staticmethod
    def _spans(labels: IntArray, axis: int) -> bool:
        """Return whether one cluster touches both opposite edges along an axis."""
        first = set(np.take(labels, 0, axis=axis).ravel()) - {0}
        last = set(np.take(labels, -1, axis=axis).ravel()) - {0}
        return bool(first & last)
