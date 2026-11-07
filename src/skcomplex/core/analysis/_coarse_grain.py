"""Coarse-grain agents into fields on a grid, or the members of a group into a density over the values of a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import SNAPSHOT_DIMENSIONS, _read_snapshots
from .types import Source

TAGS = AnalysisTags(DataKind.AGENTS)


@dataclass(frozen=True)
class CoarseGrainResult:
    """Agents coarse-grained into a field on a grid, or members into a density over the values of a state.

    Args:
        positions:
            The positions, one row per agent, with one leading axis of frames for a trajectory, or the values of the
            state of every member with `over`.

        box:
            The sides of the box, or `None` with `over`.

        field:
            The grid of counts, or of sums with the value components last, with one grid per frame for a trajectory,
            ready for field analyses, or the fractions of members at every value of `over`, one row per frame.
    """

    positions: Array
    box: Array | None
    field: Array


def coarse_grain(
    source: Source,
    *,
    shape: Sequence[int] = (32, 32),
    box: Sequence[float] | None = None,
    values: npt.ArrayLike | None = None,
    over: npt.ArrayLike | None = None,
    state: str | None = None,
) -> CoarseGrainResult:
    """Return agents coarse-grained into a field on a grid, or members into a density over the values of a state.

    An array holds one snapshot, one row per agent in the box. A trajectory gives the positions of its first agent or
    particle group at every frame, in the box of its space unless `box` is set, and one grid per frame.

    With `over`, the function coarse-grains in state space instead: every member counts at the given value nearest to
    its value of a state, and the field is the fraction of members at every value, which sums to 1. An array holds the
    values of one snapshot, one per member, or one row per frame, and a trajectory gives the values of the state `state`
    at every frame. The densities are the initial values of a distribution site group with one member per value.

    Args:
        source:
            The trajectory, or an array of positions or of values of a state.

        shape:
            The number of cells along every side.

        box:
            The sides of the box, or `None` for the box of the space of a trajectory.

        values:
            One value or one row of values per agent of an array, such as unit directions for a polarization field, or
            `None` to count agents.

        over:
            The increasing values of a state to coarse-grain the members over, or `None` to coarse-grain positions.

        state:
            The dotted name of the state whose values a trajectory gives, such as `traders.wealth`, with `over`.

    Returns:
        result:
            The positions or values, the box, and the field.

    Raises:
        AnalysisError:
            A setting is invalid or missing, or the source holds no positions or values to coarse-grain.

    Examples:
        >>> from skcomplex.core.analysis import coarse_grain
        >>> positions = [[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]]
        >>> coarse_grain(positions, shape=[2, 2], box=[1.0, 1.0]).field.tolist()
        [[2.0, 0.0], [0.0, 1.0]]
        >>> wealth = [-1.0, 0.4, 0.6, 1.5, 9.0]
        >>> coarse_grain(wealth, over=[0.0, 1.0, 2.0]).field.tolist()
        [0.4, 0.4, 0.2]
    """
    _check(shape, box, values, over, state)
    if over is not None:
        positions = _read_state(source, state)
        grid = np.asarray(over, dtype=float)
        frames = positions if positions.ndim == SNAPSHOT_DIMENSIONS else positions[None]
        densities = np.asarray([_get_density(row, grid) for row in frames])
        return CoarseGrainResult(positions, None, densities if positions.ndim == SNAPSHOT_DIMENSIONS else densities[0])
    if isinstance(source, Trajectory):
        snapshots = _read_snapshots(source)
        if snapshots.positions is None:
            error_msg = 'The agents of the trajectory have no positions with coordinates to coarse-grain.'
            raise AnalysisError(error_msg)
        positions = snapshots.positions
        sides = box if box is not None else snapshots.box
    else:
        positions = np.asarray(source, dtype=float)
        sides = box
    if sides is None:
        error_msg = 'coarse_grain needs the sides of the box. Set `box`.'
        raise AnalysisError(error_msg)
    read_box = np.asarray(sides, dtype=np.float64)
    if len(shape) != read_box.size:
        error_msg = f'The shape {tuple(shape)} needs one entry per dimension of the box, {read_box.size}.'
        raise AnalysisError(error_msg)
    if positions.ndim > SNAPSHOT_DIMENSIONS:
        field = np.asarray([_get_field(frame, shape, read_box, values) for frame in positions])
    else:
        field = _get_field(positions, shape, read_box, values)
    return CoarseGrainResult(positions, read_box, field)


def _check(
    shape: Sequence[int],
    box: Sequence[float] | None,
    values: npt.ArrayLike | None,
    over: npt.ArrayLike | None,
    state: str | None,
) -> None:
    """Validate the shape against a given box, or the values a state is coarse-grained over."""
    if over is not None:
        grid = np.asarray(over, dtype=float)
        if grid.ndim != 1 or not grid.size or np.any(np.diff(grid) <= 0):
            error_msg = f'coarse_grain has over={over!r}. Use a nonempty increasing vector of values.'
            raise AnalysisError(error_msg)
        if values is not None:
            error_msg = 'coarse_grain counts members over values of a state, so it takes no `values`.'
            raise AnalysisError(error_msg)
        return
    if state is not None:
        error_msg = f'coarse_grain has state={state!r} but no values to bin it over. Set `over`.'
        raise AnalysisError(error_msg)
    if any(int(side) < 1 for side in shape):
        error_msg = f'coarse_grain has shape={tuple(shape)!r}. Use 1 or more cells per side.'
        raise AnalysisError(error_msg)
    if box is not None and len(shape) != len(box):
        error_msg = f'The shape {tuple(shape)} needs one entry per dimension of the box, {len(box)}.'
        raise AnalysisError(error_msg)


def _read_state(source: Source, state: str | None) -> Array:
    """Return the values of the state of every member, from an array or from a trajectory."""
    if isinstance(source, Trajectory):
        if state is None:
            error_msg = 'coarse_grain reads a trajectory with `over` at the state it names. Set `state`.'
            raise AnalysisError(error_msg)
        read = np.asarray(source.values(state), dtype=float)
        positions = read.reshape(read.shape[0], -1)
    else:
        positions = np.asarray(source, dtype=float)
    if not positions.size or positions.ndim > SNAPSHOT_DIMENSIONS:
        error_msg = 'coarse_grain needs the values of one member or more, one row per frame.'
        raise AnalysisError(error_msg)
    return positions


def _get_density(values: Array, grid: Array) -> Array:
    """Return the fraction of members at the value of the grid nearest to their value, ties to the lower value."""
    if grid.size == 1:
        return np.ones(1)
    nearest = np.clip(np.searchsorted(grid, values), 1, grid.size - 1)
    nearest -= np.abs(values - grid[nearest - 1]) <= np.abs(values - grid[nearest])
    return np.bincount(nearest, minlength=grid.size) / values.size


def _get_field(positions: Array, shape: Sequence[int], box: Array, values: npt.ArrayLike | None) -> Array:
    """Return the grid of counts, or of sums of the values, of one snapshot."""
    sides = tuple(int(side) for side in shape)
    array = np.mod(positions.reshape(-1, box.size), box)
    cells = np.minimum((array / box * np.asarray(sides)).astype(int), np.asarray(sides) - 1)
    flat = np.ravel_multi_index(tuple(cells.T), sides)
    total = int(np.prod(sides))
    if values is None:
        return np.bincount(flat, minlength=total).reshape(sides).astype(float)
    weights = np.asarray(values, dtype=float).reshape(flat.size, -1)
    columns = [np.bincount(flat, weights[:, index], minlength=total) for index in range(weights.shape[1])]
    field = np.stack(columns, axis=-1).reshape(*sides, weights.shape[1])
    return field[..., 0] if weights.shape[1] == 1 else field
