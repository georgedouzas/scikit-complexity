"""Measure spatial structure: clusters on grids, spatial correlation functions, and box-counting dimensions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from ..base import AnalysisError

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
        >>> from skcomplex.core.analysis import label_clusters
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


def correlation_function(field: npt.ArrayLike, max_r: int | None = None) -> tuple[Array, Array]:
    """Return the radially averaged spatial correlation of a field on a periodic grid, computed with the FFT.

    Args:
        field:
            The values on a grid of any dimension.

        max_r:
            The largest distance in cells, or `None` for half the smallest side.

    Returns:
        correlation:
            The distances and the correlation at each of them, 1 at distance 0.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import correlation_function
        >>> distances, values = correlation_function(np.random.default_rng(0).normal(size=(64, 64)), 3)
        >>> float(values[0]), bool(abs(values[1]) < 0.1)
        (1.0, True)
    """
    grid = np.asarray(field, dtype=float)
    grid = grid - grid.mean()
    spectrum = np.fft.fftn(grid)
    correlation = np.real(np.fft.ifftn(spectrum * np.conj(spectrum))) / grid.size
    if correlation.flat[0] == 0:
        error_msg = 'The field is constant, so it has no correlation function.'
        raise AnalysisError(error_msg)
    correlation /= correlation.flat[0]
    axes = np.meshgrid(*(np.minimum(np.arange(size), size - np.arange(size)) for size in grid.shape), indexing='ij')
    distance = np.rint(np.sqrt(sum(axis**2 for axis in axes))).astype(int)
    largest = min(grid.shape) // 2 if max_r is None else max_r
    sums = np.bincount(distance.ravel(), correlation.ravel())[: largest + 1]
    counts = np.bincount(distance.ravel())[: largest + 1]
    return np.arange(largest + 1, dtype=float), sums / counts


def box_counting_dimension(occupied: npt.ArrayLike, sizes: npt.ArrayLike | None = None) -> float:
    """Return the box-counting dimension of a set of cells, the slope of the log box count against the log box size.

    Args:
        occupied:
            A boolean grid of any dimension.

        sizes:
            The box sides in cells, or `None` for powers of 2 up to an eighth of the smallest side.

    Returns:
        dimension:
            The box-counting dimension.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import box_counting_dimension
        >>> round(box_counting_dimension(np.ones((256, 256), dtype=bool)), 2)
        2.0
    """
    grid = np.asarray(occupied, dtype=bool)
    sides = 2 ** np.arange(int(np.log2(min(grid.shape) // 8)) + 1) if sizes is None else np.asarray(sizes, dtype=int)
    counts = []
    for side in sides:
        trimmed = grid[tuple(slice(0, size - size % side) for size in grid.shape)]
        shape = [item for size in trimmed.shape for item in (size // side, side)]
        boxes = trimmed.reshape(shape).any(axis=tuple(range(1, 2 * grid.ndim, 2)))
        counts.append(boxes.sum())
    return float(-np.polyfit(np.log(sides), np.log(counts), 1)[0])
