"""Coarse-grain agent snapshots into fields on a grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]


@analysis(DataKind.AGENTS)
def coarse_grain(
    positions: npt.ArrayLike,
    box: npt.ArrayLike,
    shape: Sequence[int],
    values: npt.ArrayLike | None = None,
) -> Array:
    """Return agents coarse-grained into a field on a grid: their number, or the sum of a value, in every cell.

    Args:
        positions:
            One row per agent in the box.

        box:
            The sides of the box.

        shape:
            The number of cells along every side.

        values:
            One value or one row of values per agent, such as unit headings for a polarization field, or `None` to
            count agents.

    Returns:
        field:
            The grid of counts, or of sums with the value components last, ready for field analyses.

    Raises:
        AnalysisError:
            The shape does not have one entry per dimension of the box.

    Examples:
        >>> from skcomplex.analysis import coarse_grain
        >>> coarse_grain([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]], [1.0, 1.0], [2, 2]).tolist()
        [[2.0, 0.0], [0.0, 1.0]]
    """
    sides = np.asarray(box, dtype=float)
    if len(shape) != sides.size:
        error_msg = f'The shape {tuple(shape)} needs one entry per dimension of the box, {sides.size}.'
        raise AnalysisError(error_msg)
    array = np.mod(np.asarray(positions, dtype=float).reshape(-1, sides.size), sides)
    cells = np.minimum((array / sides * np.asarray(shape)).astype(int), np.asarray(shape) - 1)
    flat = np.ravel_multi_index(tuple(cells.T), tuple(shape))
    total = int(np.prod(shape))
    if values is None:
        return np.bincount(flat, minlength=total).reshape(tuple(shape)).astype(float)
    weights = np.asarray(values, dtype=float).reshape(flat.size, -1)
    columns = [np.bincount(flat, weights[:, index], minlength=total) for index in range(weights.shape[1])]
    field = np.stack(columns, axis=-1).reshape(*shape, weights.shape[1])
    return field[..., 0] if weights.shape[1] == 1 else field
