"""Find the point defects of a two-dimensional field of axes or vectors by the winding of their angles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _read_fields
from .types import Source

FIELD_RANK = 2

CHARGE_TOLERANCE = 1e-6

TAGS = AnalysisTags(DataKind.FIELD)


@dataclass(frozen=True)
class DefectsResult:
    """Point defects of a two-dimensional field of angles, with their topological charges, at every frame.

    Args:
        angles:
            The angles, with one frame per row of the first axis.

        positions:
            The centres of the plaquettes with a defect at every frame, each an array of defects by dimensions.

        charges:
            The charges of the defects at every frame, each an array in the order of `positions`.
    """

    angles: Array
    positions: list[Array]
    charges: list[Array]


def defects(
    source: Source,
    *,
    period: float = np.pi,
    variables: Sequence[str] | None = None,
    periodic: bool = True,
) -> DefectsResult:
    """Return the point defects of a two-dimensional field of angles, with their topological charges, at every frame.

    The winding around a plaquette of four neighbouring sites is the sum of the differences of the angles along its
    edges, each wrapped to one period around 0, over `2 pi`. A plaquette with a winding that is not 0 holds a defect of
    that charge. The charges of a periodic lattice sum to 0.

    An array of two dimensions holds one snapshot, and an array of three dimensions one frame per row of its first axis.

    Args:
        source:
            The trajectory, or an array of angles of one snapshot or of one frame per row.

        period:
            The period of the angles, `np.pi` for axes without head and tail and `2 * np.pi` for vectors.

        variables:
            The field of a trajectory to read, or `None` for its first field.

        periodic:
            Whether the lattice joins its opposite edges, or has plaquettes only between its sites.

    Returns:
        result:
            The angles, and the positions and the charges of the defects of every frame.

    Raises:
        AnalysisError:
            The period is not positive, or the field is not two-dimensional.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import defects
        >>> rows, columns = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing='ij')
        >>> vortex = np.arctan2(rows - 3.5, columns - 3.5)
        >>> result = defects(vortex, period=2 * np.pi, periodic=False)
        >>> result.positions[0].tolist(), result.charges[0].tolist()
        ([[3.5, 3.5]], [1.0])
    """
    if not period > 0:
        error_msg = f'The period of the angles is {period!r}. Give a positive number.'
        raise AnalysisError(error_msg)
    frames = _read_fields(source, variables[0] if variables else None)
    if frames.ndim not in {FIELD_RANK, FIELD_RANK + 1}:
        error_msg = f'defects reads a two-dimensional field, but got the shape {frames.shape}.'
        raise AnalysisError(error_msg)
    angles = frames[None] if frames.ndim == FIELD_RANK else frames
    positions, charges = [], []
    for frame in angles:
        windings = _windings(frame, period, periodic=periodic)
        found = np.argwhere(np.abs(windings) > CHARGE_TOLERANCE)
        positions.append(np.asarray(found + 0.5, dtype=float))
        charges.append(np.round(windings[tuple(found.T)], 6))
    return DefectsResult(angles, positions, charges)


def _wrapped(differences: Array, period: float) -> Array:
    """Return differences of angles in one period around 0."""
    half = period / 2
    return np.mod(differences + half, period) - half


def _windings(angles: Array, period: float, *, periodic: bool) -> Array:
    """Return the winding around every plaquette of a frame, in units of a full turn."""
    below = np.roll(angles, -1, axis=0)
    right = np.roll(angles, -1, axis=1)
    diagonal = np.roll(below, -1, axis=1)
    steps = [right - angles, diagonal - right, below - diagonal, angles - below]
    windings = sum(_wrapped(step, period) for step in steps) / (2 * np.pi)
    return np.asarray(windings) if periodic else np.asarray(windings)[:-1, :-1]
