"""Find the point defects of a two-dimensional field of axes or vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..enums import DataKind
from ..errors import AnalysisError
from ..types import Array
from ._analysis_tags import AnalysisTags
from ._base import _read_fields
from .types import DataSource

FIELD_RANK = 2
CHARGE_TOLERANCE = 1e-6
TAGS = AnalysisTags(DataKind.FIELD)


@dataclass(frozen=True)
class DefectsResult:
    """A result of `defects`, the points where a field of angles winds around.

    It holds, for every frame, the angles, the positions of the defects, and their charges, the number of whole turns
    the angle makes around each, such as one defect of charge 1 at the centre of a vortex. It reports the vortices of a
    superfluid, the defects of a liquid crystal, and the swirls of a swarm of bacteria.

    Read more in the user guide, under Analyse.

    Args:
        angles:
            The angles, with one frame per row of the first axis.

        positions:
            The centres of the plaquettes with a defect at every frame, each an array of defects by dimensions.

        charges:
            The charges of the defects at every frame, each an array in the order of `positions`.

    See Also:
        defects: Returns this result.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import defects
        >>> rows, columns = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing='ij')
        >>> result = defects(np.arctan2(rows - 3.5, columns - 3.5), period=2 * np.pi, periodic=False)
        >>> result.positions[0].tolist(), result.charges[0].tolist()
        ([[3.5, 3.5]], [1.0])
    """

    angles: Array
    positions: list[Array]
    charges: list[Array]


def defects(
    source: DataSource,
    *,
    period: float = np.pi,
    variables: Sequence[str] | None = None,
    periodic: bool = True,
) -> DefectsResult:
    """Return the points where a field of angles winds around, at every frame.

    Around every square of four neighbouring sites it adds the changes of the angle along the sides, each wrapped to
    within half a turn, and divides by a whole turn, so a square the angle winds around once holds a defect of charge 1.
    The change along every edge is wrapped once, so the charges of a periodic grid always sum to 0, even where a change
    is exactly half a turn. An array of two dimensions holds one frame and one of three dimensions one frame per row.
    It finds the vortices of a superfluid, the defects of a liquid crystal, and the swirls of a swarm of
    bacteria.

    Read more in the user guide, under Analyse.

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

    See Also:
        alignment: Measures how well the members line up.
        domains: Finds the regions of a field.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import defects
        >>> rows, columns = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing='ij')
        >>> result = defects(np.arctan2(rows - 3.5, columns - 3.5), period=2 * np.pi, periodic=False)
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
    """Return the winding around every plaquette of a frame, in units of a full turn.

    The change along every edge is wrapped once, from a site to its right or lower neighbour, and a plaquette that walks
    the edge the other way takes its negative, so a change of exactly half a turn cancels between two plaquettes.
    """
    across = _wrapped(np.roll(angles, -1, axis=1) - angles, period)
    down = _wrapped(np.roll(angles, -1, axis=0) - angles, period)
    windings = (across + np.roll(down, -1, axis=1) - np.roll(across, -1, axis=0) - down) / (2 * np.pi)
    return np.asarray(windings) if periodic else np.asarray(windings)[:-1, :-1]
