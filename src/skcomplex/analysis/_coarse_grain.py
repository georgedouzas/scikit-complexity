"""Coarse-grain agents into fields on a grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]


class CoarseGrainAnalysis(BaseAnalysis):
    """Agents coarse-grained into a field on a grid: their number, or the sum of a value, in every cell.

    An array holds one snapshot, one row per agent in the box. A trajectory gives the positions of its first agent or
    particle group at every frame, in the box of its space unless `box` is set, and one grid per frame.

    Args:
        shape:
            The number of cells along every side.

        box:
            The sides of the box, or `None` for the box of the space of a trajectory.

        values:
            One value or one row of values per agent of an array, such as unit headings for a polarization field, or
            `None` to count agents.

    Attributes:
        positions_:
            The positions, one row per agent, with one leading axis of frames for a trajectory.

        box_:
            The sides of the box.

        field_:
            The grid of counts, or of sums with the value components last, with one grid per frame for a trajectory,
            ready for field analyses.

    Examples:
        >>> from skcomplex.analysis import CoarseGrainAnalysis
        >>> positions = [[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]]
        >>> CoarseGrainAnalysis([2, 2], box=[1.0, 1.0]).check().prepare(positions).run().field_.tolist()
        [[2.0, 0.0], [0.0, 1.0]]
    """

    def __init__(
        self,
        shape: Sequence[int] = (32, 32),
        *,
        box: Sequence[float] | None = None,
        values: npt.ArrayLike | None = None,
    ) -> None:
        self.shape = shape
        self.box = box
        self.values = values

    def _check(self) -> Self:
        """Validate that the shape has positive sides, one per dimension of a given box."""
        if any(int(side) < 1 for side in self.shape):
            error_msg = f'{type(self).__name__} has shape={tuple(self.shape)!r}. Use 1 or more cells per side.'
            raise AnalysisError(error_msg)
        if self.box is not None and len(self.shape) != len(self.box):
            error_msg = f'The shape {tuple(self.shape)} needs one entry per dimension of the box, {len(self.box)}.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the agents kind."""
        return DataKind.AGENTS

    def _prepare(self, source: Source) -> Self:
        """Read the positions of an array, or of the agents of a trajectory with the box of its space."""
        if isinstance(source, Trajectory):
            snapshots = self._read_snapshots(source)
            if snapshots.positions is None:
                error_msg = 'The agents of the trajectory have no positions with coordinates to coarse-grain.'
                raise AnalysisError(error_msg)
            self.positions_ = snapshots.positions
            box = self.box if self.box is not None else snapshots.box
        else:
            self.positions_ = np.asarray(source, dtype=float)
            box = self.box
        if box is None:
            error_msg = f'{type(self).__name__} needs the sides of the box. Set `box`.'
            raise AnalysisError(error_msg)
        self.box_ = np.asarray(box, dtype=float)
        if len(self.shape) != self.box_.size:
            error_msg = f'The shape {tuple(self.shape)} needs one entry per dimension of the box, {self.box_.size}.'
            raise AnalysisError(error_msg)
        return self

    def _get_field(self, positions: Array) -> Array:
        """Return the grid of counts, or of sums of the values, of one snapshot."""
        shape = tuple(int(side) for side in self.shape)
        array = np.mod(positions.reshape(-1, self.box_.size), self.box_)
        cells = np.minimum((array / self.box_ * np.asarray(shape)).astype(int), np.asarray(shape) - 1)
        flat = np.ravel_multi_index(tuple(cells.T), shape)
        total = int(np.prod(shape))
        if self.values is None:
            return np.bincount(flat, minlength=total).reshape(shape).astype(float)
        weights = np.asarray(self.values, dtype=float).reshape(flat.size, -1)
        columns = [np.bincount(flat, weights[:, index], minlength=total) for index in range(weights.shape[1])]
        field = np.stack(columns, axis=-1).reshape(*shape, weights.shape[1])
        return field[..., 0] if weights.shape[1] == 1 else field

    def _run(self) -> Self:
        """Compute the field of the snapshot, or of every frame of a trajectory."""
        if self.positions_.ndim > 2:  # noqa: PLR2004
            self.field_ = np.asarray([self._get_field(frame) for frame in self.positions_])
            return self
        self.field_ = self._get_field(self.positions_)
        return self
