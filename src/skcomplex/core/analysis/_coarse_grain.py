"""Coarse-grain agents into fields, members into densities, or lattices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine.runtime import Trajectory
from ..enums import DataKind
from ..errors import AnalysisError
from ..types import Array
from ._analysis_tags import AnalysisTags
from ._base import SNAPSHOT_DIMENSIONS, _read_fields, _read_snapshots
from .types import DataSource

TAGS = AnalysisTags(DataKind.AGENTS)
RULES = ('majority', 'mean')


@dataclass(frozen=True)
class CoarseGrainResult:
    """A result of `coarse_grain`, agents on a grid, members over values, or blocks.

    It holds the counts of agents in every cell of a grid, one grid per frame, with the positions and the box, the
    share of members at every value, which sums to 1, or the value of every block of a lattice. It reports the density
    of a herd, the spread of wealth, the density of a fluid, and the block spins of a magnet.

    Read more in the user guide, under Analyse.

    Args:
        positions:
            The positions, one row per agent, with one leading axis of frames for a trajectory, the values of the
            state of every member with `over`, or the lattice with `blocks`.

        box:
            The sides of the box, or `None` with `over` or `blocks`.

        field:
            The grid of counts, or of sums with the value components last, with one grid per frame for a trajectory,
            ready for field analyses, the fractions of members at every value of `over`, one row per frame, or the
            lattice of blocks with `blocks`, one per frame for a trajectory.

    See Also:
        coarse_grain: Returns this result.

    Examples:
        >>> from skcomplex.core.analysis import coarse_grain
        >>> coarse_grain([-1.0, 0.4, 0.6, 1.5, 9.0], over=[0.0, 1.0, 2.0]).field.tolist()
        [0.4, 0.4, 0.2]
    """

    positions: Array
    box: Array | None
    field: Array


def coarse_grain(
    source: DataSource,
    *,
    shape: Sequence[int] = (32, 32),
    box: Sequence[float] | None = None,
    values: npt.ArrayLike | None = None,
    over: npt.ArrayLike | None = None,
    state: str | None = None,
    blocks: int | Sequence[int] | None = None,
    rule: str = 'majority',
    tie: float | str = 'random',
    random_state: int | np.random.Generator | None = None,
) -> CoarseGrainResult:
    """Return agents on a grid, members over values of a state, or lattice blocks.

    It counts the agents in every cell of a grid of `shape` over the box, one grid per frame, such as two agents in one
    corner and one in the other of a box of side 1. With `over`, it counts every member at the given value nearest to
    its own instead, and returns the share at every value. An array holds one snapshot, one row per agent, and a
    trajectory gives the first group at every frame. With `blocks`, it replaces every block of a lattice by the value
    most of its sites hold, or by their mean, as the block spins of a renormalization group step. It turns animals
    into the density of a herd, traders into the spread of wealth, particles into the density of a fluid, and the spins
    of a magnet into block spins.

    Read more in the user guide, under Analyse.

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
            The dotted name of the state whose values a trajectory gives, such as `traders.wealth`, with `over`, or the
            lattice state with `blocks`, `None` for the first one.

        blocks:
            The side of the blocks, one for every axis or one per axis, or `None` to coarse-grain agents or members.
            Every side of the lattice must be a multiple of it.

        rule:
            `'majority'` for the value most sites of a block hold, or `'mean'` for their mean, with `blocks`.

        tie:
            The value of a block whose most frequent values tie under `'majority'`, or `'random'` to draw one of them.

        random_state:
            The seed or generator of the draws of ties.

    Returns:
        result:
            The positions or values, the box, and the field.

    Raises:
        AnalysisError:
            A setting is invalid or missing, or the source holds no positions or values to coarse-grain.

    See Also:
        distribution: Fits and tests the distribution of values.
        clusters: Finds the clusters of a grid.

    Examples:
        >>> from skcomplex.core.analysis import coarse_grain
        >>> positions = [[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]]
        >>> coarse_grain(positions, shape=[2, 2], box=[1.0, 1.0]).field.tolist()
        [[2.0, 0.0], [0.0, 1.0]]
    """
    if blocks is not None:
        return _coarse_grain_blocks(
            source,
            blocks,
            rule=rule,
            tie=tie,
            random_state=random_state,
            state=state,
            over=over,
            values=values,
        )
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


def _coarse_grain_blocks(
    source: DataSource,
    blocks: int | Sequence[int],
    *,
    rule: str,
    tie: float | str,
    random_state: int | np.random.Generator | None,
    state: str | None,
    over: npt.ArrayLike | None,
    values: npt.ArrayLike | None,
) -> CoarseGrainResult:
    """Return a lattice coarse-grained over blocks, for one snapshot or every frame."""
    if over is not None or values is not None:
        error_msg = 'coarse_grain over blocks reads the values of a lattice, so it takes no `over` or `values`.'
        raise AnalysisError(error_msg)
    if rule not in RULES:
        error_msg = f'coarse_grain has rule={rule!r}. Use one of {", ".join(RULES)}.'
        raise AnalysisError(error_msg)
    if isinstance(tie, str) and tie != 'random':
        error_msg = f"coarse_grain has tie={tie!r}. Use 'random' or a value."
        raise AnalysisError(error_msg)
    lattice = _read_fields(source, state)
    frames = lattice if isinstance(source, Trajectory) else lattice[None]
    sides = frames.shape[1:]
    sizes = (blocks,) * len(sides) if isinstance(blocks, int) else tuple(blocks)
    if len(sizes) != len(sides) or any(size < 1 or side % size for side, size in zip(sides, sizes, strict=False)):
        error_msg = (
            f'coarse_grain has blocks={blocks!r} for a lattice of sides {sides}. Use blocks whose sides divide those of'
            ' the lattice, one for every axis or one per axis.'
        )
        raise AnalysisError(error_msg)
    split = [length for side, size in zip(sides, sizes, strict=True) for length in (side // size, size)]
    grouped = frames.reshape(len(frames), *split)
    axes = tuple(range(2, grouped.ndim, 2))
    if rule == 'mean':
        field = grouped.mean(axis=axes)
    else:
        field = _get_majority(grouped, axes, tie, np.random.default_rng(random_state))
    return CoarseGrainResult(lattice, None, field if isinstance(source, Trajectory) else field[0])


def _get_majority(grouped: Array, axes: tuple[int, ...], tie: float | str, rng: np.random.Generator) -> Array:
    """Return the value most sites of every block hold.

    A tie between the most frequent values takes `tie`, or one of them drawn at random.
    """
    candidates = np.unique(grouped)
    counts = np.stack([(grouped == value).sum(axis=axes) for value in candidates], axis=-1)
    best = counts == counts.max(axis=-1, keepdims=True)
    tied = best.sum(axis=-1) > 1
    noise = rng.random(best.shape) if tie == 'random' else np.zeros(best.shape)
    field = candidates[np.argmax(np.where(best, 1.0 + noise, 0.0), axis=-1)].astype(float)
    if tie != 'random':
        field[tied] = float(tie)
    return field


def _check(
    shape: Sequence[int],
    box: Sequence[float] | None,
    values: npt.ArrayLike | None,
    over: npt.ArrayLike | None,
    state: str | None,
) -> None:
    """Validate the shape against a given box, or the values of a coarse-grained state."""
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


def _read_state(source: DataSource, state: str | None) -> Array:
    """Return the values of the state of every member.

    They come from an array or from a trajectory.
    """
    if isinstance(source, Trajectory):
        if state is None:
            error_msg = 'coarse_grain reads a trajectory with `over` at the state it names. Set `state`.'
            raise AnalysisError(error_msg)
        read = np.asarray(source.get_values(state), dtype=float)
        positions = read.reshape(read.shape[0], -1)
    else:
        positions = np.asarray(source, dtype=float)
    if not positions.size or positions.ndim > SNAPSHOT_DIMENSIONS:
        error_msg = 'coarse_grain needs the values of one member or more, one row per frame.'
        raise AnalysisError(error_msg)
    return positions


def _get_density(values: Array, grid: Array) -> Array:
    """Return the fraction of members at the value of the grid nearest to their value.

    Ties go to the lower value.
    """
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
