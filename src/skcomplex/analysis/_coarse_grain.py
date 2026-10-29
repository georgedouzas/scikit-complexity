"""Coarse-grain agents into fields on a grid, or the members of a group into a density over the values of a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import SNAPSHOT_DIMENSIONS, BaseAnalysis
from .types import Source


class CoarseGrainAnalysis(BaseAnalysis):
    """Agents coarse-grained into a field on a grid, or members into a density over the values of a state.

    An array holds one snapshot, one row per agent in the box. A trajectory gives the positions of its first agent or
    particle group at every frame, in the box of its space unless `box` is set, and one grid per frame.

    With `over`, the analysis coarse-grains in state space instead: every member counts at the given value nearest to
    its value of a state, and the field is the fraction of members at every value, which sums to 1. An array holds the
    values of one snapshot, one per member, or one row per frame, and a trajectory gives the values of the state `state`
    at every frame. The densities are the initial values of a distribution site group with one member per value.

    Args:
        shape:
            The number of cells along every side.

        box:
            The sides of the box, or `None` for the box of the space of a trajectory.

        values:
            One value or one row of values per agent of an array, such as unit headings for a polarization field, or
            `None` to count agents.

        over:
            The increasing values of a state to coarse-grain the members over, or `None` to coarse-grain positions.

        state:
            The dotted name of the state whose values a trajectory gives, such as `traders.wealth`, with `over`.

    Attributes:
        positions_:
            The positions, one row per agent, with one leading axis of frames for a trajectory, or the values of the
            state of every member with `over`.

        box_:
            The sides of the box, or `None` with `over`.

        field_:
            The grid of counts, or of sums with the value components last, with one grid per frame for a trajectory,
            ready for field analyses, or the fractions of members at every value of `over`, one row per frame.

    Examples:
        >>> from skcomplex.analysis import CoarseGrainAnalysis
        >>> positions = [[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]]
        >>> CoarseGrainAnalysis([2, 2], box=[1.0, 1.0]).check().prepare(positions).run().field_.tolist()
        [[2.0, 0.0], [0.0, 1.0]]
        >>> wealth = [-1.0, 0.4, 0.6, 1.5, 9.0]
        >>> CoarseGrainAnalysis(over=[0.0, 1.0, 2.0]).check().prepare(wealth).run().field_.tolist()
        [0.4, 0.4, 0.2]
    """

    def __init__(
        self,
        shape: Sequence[int] = (32, 32),
        *,
        box: Sequence[float] | None = None,
        values: npt.ArrayLike | None = None,
        over: npt.ArrayLike | None = None,
        state: str | None = None,
    ) -> None:
        self.shape = shape
        self.box = box
        self.values = values
        self.over = over
        self.state = state

    def _check(self) -> Self:
        """Validate the shape against a given box, or the values a state is coarse-grained over."""
        if self.over is not None:
            grid = np.asarray(self.over, dtype=float)
            if grid.ndim != 1 or not grid.size or np.any(np.diff(grid) <= 0):
                error_msg = f'{type(self).__name__} has over={self.over!r}. Use a nonempty increasing vector of values.'
                raise AnalysisError(error_msg)
            if self.values is not None:
                error_msg = f'{type(self).__name__} counts members over values of a state, so it takes no `values`.'
                raise AnalysisError(error_msg)
            return self
        if self.state is not None:
            error_msg = f'{type(self).__name__} has state={self.state!r} but no values to bin it over. Set `over`.'
            raise AnalysisError(error_msg)
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
        if self.over is not None:
            return self._prepare_state(source)
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
        self.box_: Array | None = np.asarray(box, dtype=np.float64)
        if len(self.shape) != self.box_.size:
            error_msg = f'The shape {tuple(self.shape)} needs one entry per dimension of the box, {self.box_.size}.'
            raise AnalysisError(error_msg)
        return self

    def _prepare_state(self, source: Source) -> Self:
        """Read the values of the state of every member, from an array or from a trajectory."""
        if isinstance(source, Trajectory):
            if self.state is None:
                error_msg = f'{type(self).__name__} reads a trajectory with `over` at the state it names. Set `state`.'
                raise AnalysisError(error_msg)
            values = np.asarray(source.values(self.state), dtype=float)
            self.positions_ = values.reshape(values.shape[0], -1)
        else:
            self.positions_ = np.asarray(source, dtype=float)
        if not self.positions_.size or self.positions_.ndim > SNAPSHOT_DIMENSIONS:
            error_msg = f'{type(self).__name__} needs the values of one member or more, one row per frame.'
            raise AnalysisError(error_msg)
        self.box_ = None
        return self

    def _get_density(self, values: Array) -> Array:
        """Return the fraction of members at the value of `over` nearest to their value, ties to the lower value."""
        grid = np.asarray(self.over, dtype=float)
        if grid.size == 1:
            return np.ones(1)
        nearest = np.clip(np.searchsorted(grid, values), 1, grid.size - 1)
        nearest -= np.abs(values - grid[nearest - 1]) <= np.abs(values - grid[nearest])
        return np.bincount(nearest, minlength=grid.size) / values.size

    def _get_field(self, positions: Array) -> Array:
        """Return the grid of counts, or of sums of the values, of one snapshot."""
        if self.box_ is None:
            error_msg = f'{type(self).__name__} counts positions in a box, but read the values of a state with `over`.'
            raise AnalysisError(error_msg)
        shape = tuple(int(side) for side in self.shape)
        box = self.box_
        array = np.mod(positions.reshape(-1, box.size), box)
        cells = np.minimum((array / box * np.asarray(shape)).astype(int), np.asarray(shape) - 1)
        flat = np.ravel_multi_index(tuple(cells.T), shape)
        total = int(np.prod(shape))
        if self.values is None:
            return np.bincount(flat, minlength=total).reshape(shape).astype(float)
        weights = np.asarray(self.values, dtype=float).reshape(flat.size, -1)
        columns = [np.bincount(flat, weights[:, index], minlength=total) for index in range(weights.shape[1])]
        field = np.stack(columns, axis=-1).reshape(*shape, weights.shape[1])
        return field[..., 0] if weights.shape[1] == 1 else field

    def _run(self, inputs: object = None) -> Self:
        """Compute the field of the snapshot, or of every frame of a trajectory."""
        if self.over is not None:
            frames = self.positions_ if self.positions_.ndim == SNAPSHOT_DIMENSIONS else self.positions_[None]
            densities = np.asarray([self._get_density(row) for row in frames])
            self.field_ = densities if self.positions_.ndim == SNAPSHOT_DIMENSIONS else densities[0]
            return self
        if self.positions_.ndim > SNAPSHOT_DIMENSIONS:
            self.field_ = np.asarray([self._get_field(frame) for frame in self.positions_])
            return self
        self.field_ = self._get_field(self.positions_)
        return self
