"""Measure clusters and percolation: clusters of grids and graphs, the largest cluster, and spanning."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import ndimage

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import DataKind, IntArray
from ._base import BaseAnalysis
from .types import Source

KINDS = ('labels', 'sizes', 'percolation', 'spanning')
SETTINGS = {
    'labels': {'periodic', 'diagonal', 'axis'},
    'sizes': {'periodic', 'diagonal'},
    'percolation': {'diagonal', 'axis'},
    'spanning': {'diagonal', 'axis'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


class ClusterAnalysis(BaseAnalysis):
    """Clusters of the occupied cells of a grid, or the connected components of a graph, of one kind.

    The grid is a boolean array of any dimension, the last frame of a field of a trajectory, or a graph, the last
    network of a trajectory. Cells are connected through shared faces, or also corners with `diagonal=True`.

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

    Attributes:
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
    """

    def __init__(
        self,
        kind: str = 'labels',
        variables: Sequence[str] | None = None,
        *,
        periodic: bool = False,
        diagonal: bool = False,
        axis: int = 0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.periodic = periodic
        self.diagonal = diagonal
        self.axis = axis

    def _check(self) -> Self:
        """Validate the kind and that the kind reads every changed setting."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the field kind."""
        return DataKind.FIELD

    def _prepare(self, source: Source) -> Self:
        """Read a graph, or the grids of the source, the last frame of a trajectory but for spanning."""
        self.grids_: npt.NDArray[np.bool_] | None = None
        self.graph_: nx.Graph | None = None
        if isinstance(source, nx.Graph):
            if self.kind in {'labels', 'spanning'}:
                error_msg = (
                    f'{type(self).__name__} of kind {self.kind!r} reads grids, not graphs. Use the kind sizes or '
                    'percolation.'
                )
                raise AnalysisError(error_msg)
            self.graph_ = source
            return self
        name = self.variables[0] if self.variables else None
        grids = np.asarray(self._read_fields(source, name), dtype=bool)
        if self.kind == 'spanning':
            self.grids_ = grids
        else:
            self.grids_ = grids[-1:] if isinstance(source, Trajectory) else grids[None]
        return self

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

    def _run(self, inputs: object = None) -> Self:
        """Compute the clusters, their sizes, the percolation observables, or the spanning probability."""
        if self.kind == 'labels':
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
        return self

    @staticmethod
    def _merge(labels: IntArray, axis: int) -> None:
        """Merge clusters that touch across the periodic edges of an axis, in place."""
        parent = np.arange(labels.max() + 1)

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = int(parent[item])
            return item

        for first, last in zip(
            np.take(labels, 0, axis=axis).ravel(),
            np.take(labels, -1, axis=axis).ravel(),
            strict=True,
        ):
            if first and last:
                parent[find(int(first))] = find(int(last))
        roots = np.asarray([find(int(item)) for item in range(parent.size)])
        labels[...] = roots[labels]

    @staticmethod
    def _label(grid: npt.ArrayLike, *, periodic: bool, diagonal: bool) -> tuple[IntArray, IntArray]:
        """Return the label of each cell, 0 for empty cells and 1 onwards for clusters, and the size of each cluster."""
        occupied = np.asarray(grid, dtype=bool)
        structure = ndimage.generate_binary_structure(occupied.ndim, occupied.ndim if diagonal else 1)
        labels, _ = ndimage.label(occupied, structure=structure)
        if periodic:
            for axis in range(occupied.ndim):
                ClusterAnalysis._merge(labels, axis)
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
