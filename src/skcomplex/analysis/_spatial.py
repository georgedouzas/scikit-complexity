"""Measure spatial structure: clusters on grids."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Clusters:
    """Connected clusters of occupied cells.

    Args:
        labels:
            The label of every cell, 0 for empty cells and 1 onwards for clusters.

        sizes:
            The number of cells of every cluster, in label order.
    """

    labels: npt.NDArray[np.int64]
    sizes: npt.NDArray[np.int64]

    def spans(self, axis: int = 0) -> bool:
        """Check whether one cluster touches both opposite edges along an axis, as in percolation.

        Args:
            axis:
                The axis whose two edges a spanning cluster joins.

        Returns:
            spanning:
                Whether a cluster spans the axis.
        """
        first = set(np.take(self.labels, 0, axis=axis).ravel()) - {0}
        last = set(np.take(self.labels, -1, axis=axis).ravel()) - {0}
        return bool(first & last)


def _merge(labels: npt.NDArray[np.int64], axis: int) -> None:
    """Merge clusters that touch across the periodic edges of an axis, in place."""
    parent = np.arange(labels.max() + 1)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for first, last in zip(np.take(labels, 0, axis=axis).ravel(), np.take(labels, -1, axis=axis).ravel(), strict=True):
        if first and last:
            parent[find(int(first))] = find(int(last))
    roots = np.asarray([find(int(item)) for item in range(parent.size)])
    labels[...] = roots[labels]


@analysis(DataKind.FIELD)
def label_clusters(occupied: npt.ArrayLike, *, periodic: bool = False, diagonal: bool = False) -> Clusters:
    """Return the clusters of occupied cells of a grid, connected through shared faces or also corners.

    Args:
        occupied:
            A boolean grid of any dimension.

        periodic:
            Whether clusters wrap around the edges.

        diagonal:
            Whether cells that share only a corner are connected.

    Returns:
        clusters:
            The labels and the sizes of the clusters.

    Examples:
        >>> from skcomplex.analysis import label_clusters
        >>> grid = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
        >>> label_clusters(grid).sizes.tolist(), label_clusters(grid, periodic=True).sizes.tolist()
        ([1, 1, 1, 1], [4])
    """
    grid = np.asarray(occupied, dtype=bool)
    structure = ndimage.generate_binary_structure(grid.ndim, grid.ndim if diagonal else 1)
    labels, _ = ndimage.label(grid, structure=structure)
    if periodic:
        for axis in range(grid.ndim):
            _merge(labels, axis)
    distinct = np.unique(labels[labels > 0])
    relabelled = np.zeros_like(labels)
    relabelled[labels > 0] = np.searchsorted(distinct, labels[labels > 0]) + 1
    return Clusters(relabelled, np.bincount(relabelled.ravel())[1:])
