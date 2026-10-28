"""Measure spatial structure: spatial correlation functions and box-counting dimensions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ._kinds import DataKind, analysis
from ._utils import real_array

type Array = npt.NDArray[np.float64]


@analysis(DataKind.FIELD)
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
        >>> from skcomplex.analysis import correlation_function
        >>> distances, values = correlation_function(np.random.default_rng(0).normal(size=(64, 64)), 3)
        >>> float(values[0]), bool(abs(values[1]) < 0.1)
        (1.0, True)
    """
    grid = real_array(field, 'field')
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


@analysis(DataKind.FIELD)
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
        >>> from skcomplex.analysis import box_counting_dimension
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
