"""Measure the geometry of chains of points: radius of gyration, end-to-end distance, and deviation from a reference."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import _check_choice, _read_snapshots
from .types import Source

SHAPE_KINDS = ('gyration', 'end-to-end', 'rmsd')

FRAME_DIMENSIONS = 3


@dataclass(frozen=True)
class ChainShapeResult:
    """Geometry of a chain of points per frame.

    Args:
        positions:
            The positions, frames by points by dimensions.

        values:
            The value per frame.
    """

    positions: Array
    values: Array


def chain_shape(
    source: Source,
    *,
    kind: str = 'gyration',
    group: str | None = None,
    reference: npt.ArrayLike | None = None,
) -> ChainShapeResult:
    """Return the geometry of a chain of points per frame, from the positions of the members of a group in member order.

    Args:
        source:
            The trajectory, or an array of frames by points by dimensions.

        kind:
            `'gyration'` for the radius of gyration, `'end-to-end'` for the distance between the first and the last
            point, or `'rmsd'` for the root mean square deviation from `reference` after optimal superposition.

        group:
            The path of an agent or particle group of a trajectory, or the dotted name of a lattice state whose last
            axis holds the coordinates of a point, or `None` for the first agent or particle group.

        reference:
            The positions the deviation compares with, points by dimensions, for the kind `'rmsd'`.

    Returns:
        result:
            The positions and the value of every frame.

    Raises:
        AnalysisError:
            The kind is unknown, the kind `'rmsd'` has no reference, or the positions are not frames by points by
            dimensions.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import chain_shape
        >>> chain = np.array([[[0.0, 0.0], [3.0, 4.0]]])
        >>> chain_shape(chain, kind='end-to-end').values.tolist()
        [5.0]
    """
    _check_choice('chain_shape', 'kind', kind, SHAPE_KINDS)
    if kind == 'rmsd' and reference is None:
        error_msg = "chain_shape(kind='rmsd') compares with a reference. Pass `reference`."
        raise AnalysisError(error_msg)
    read = _read_positions(source, group) if isinstance(source, Trajectory) else source
    positions: Array = np.asarray(read, dtype=float)
    if positions.ndim != FRAME_DIMENSIONS:
        error_msg = f'chain_shape reads frames by points by dimensions, not the shape {positions.shape}.'
        raise AnalysisError(error_msg)
    centred = positions - positions.mean(axis=1, keepdims=True)
    if kind == 'gyration':
        values = np.sqrt(np.mean(np.sum(centred**2, axis=2), axis=1))
    elif kind == 'end-to-end':
        values = np.linalg.norm(positions[:, -1] - positions[:, 0], axis=1)
    else:
        target = np.asarray(reference, dtype=float)
        target = target - target.mean(axis=0)
        values = np.array([_deviation(frame, target) for frame in centred])
    return ChainShapeResult(positions, values)


def _read_positions(source: Trajectory, group: str | None) -> npt.ArrayLike:
    """Return the positions of a lattice state or of a group of a trajectory."""
    if group is not None and source.problem.states.has(group):
        return source.lattice(group)
    positions = _read_snapshots(source, group).positions
    if positions is None:
        error_msg = f'The members of {group or "the first group"} have no positions.'
        raise AnalysisError(error_msg)
    return positions


def _deviation(frame: Array, reference: Array) -> float:
    """Return the root mean square deviation of centred points from a centred reference after superposition."""
    rotation, _ = Rotation.align_vectors(reference, frame)
    return float(np.sqrt(np.mean(np.sum((rotation.apply(frame) - reference) ** 2, axis=1))))
