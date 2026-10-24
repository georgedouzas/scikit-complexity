"""Measure first passages, returns, laminar phases, and absorption of paths."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
DIRECTIONS = ('up', 'down')


def _rows(paths: npt.ArrayLike) -> Array:
    """Return paths as one row per path."""
    array = np.asarray(paths, dtype=float)
    return array.reshape(1, -1) if array.ndim == 1 else array


def _clock(times: npt.ArrayLike | None, size: int) -> Array:
    """Return the times of the steps of paths, the step indices without times."""
    return np.arange(size, dtype=float) if times is None else np.asarray(times, dtype=float)


def _first_hits(hits: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    """Return the first step at which every row hits, or -1 for rows that never hit."""
    first = hits.argmax(axis=1)
    return np.where(hits.any(axis=1), first, -1)


@analysis(DataKind.PATH)
def first_passage_times(
    paths: npt.ArrayLike,
    threshold: float | None = None,
    *,
    states: Sequence[float] | None = None,
    times: npt.ArrayLike | None = None,
    direction: str = 'up',
) -> Array:
    """Return the first time every path reaches a threshold or enters a set of states.

    Args:
        paths:
            One path, or one row per path, on common times.

        threshold:
            The level to reach, from below for `'up'` and from above for `'down'`.

        states:
            The states whose first visit ends the passage, instead of a threshold.

        times:
            The common times of the steps, or `None` for step indices.

        direction:
            `'up'` to reach values at or above the threshold, `'down'` at or below it.

    Returns:
        times:
            The time from the start to the first passage of every path, `nan` for paths that never pass.

    Raises:
        AnalysisError:
            Neither a threshold nor states are given, or the direction is unknown.

    Examples:
        >>> from skcomplex.core.analysis import first_passage_times
        >>> first_passage_times([[0, 1, 2, 3], [0, 0, 0, 0]], 2).tolist()
        [2.0, nan]
    """
    array = _rows(paths)
    if direction not in DIRECTIONS:
        error_msg = f'Unknown direction {direction!r}. Use one of {", ".join(DIRECTIONS)}.'
        raise AnalysisError(error_msg)
    if states is not None:
        hits = np.isin(array, np.asarray(states, dtype=float))
    elif threshold is not None:
        hits = array >= threshold if direction == 'up' else array <= threshold
    else:
        error_msg = 'Give a threshold or the states that end the passage.'
        raise AnalysisError(error_msg)
    clock = _clock(times, array.shape[1])
    first = _first_hits(hits)
    return np.where(first >= 0, clock[np.maximum(first, 0)] - clock[0], np.nan)


@analysis(DataKind.PATH)
def hitting_probabilities(paths: npt.ArrayLike, states: Sequence[float]) -> dict[float, float]:
    """Return the share of paths that reach each of a set of absorbing states first.

    Args:
        paths:
            One path, or one row per path.

        states:
            The absorbing states.

    Returns:
        probabilities:
            The share of paths that hit every state before the others, the rest never hitting any.

    Examples:
        >>> from skcomplex.core.analysis import hitting_probabilities
        >>> hitting_probabilities([[1, 2, 3], [1, 0, 0], [1, 2, 2]], [0, 3])
        {0: 0.3333333333333333, 3: 0.3333333333333333}
    """
    array = _rows(paths)
    first = _first_hits(np.isin(array, np.asarray(states, dtype=float)))
    reached = array[np.arange(array.shape[0]), np.maximum(first, 0)]
    return {state: float(np.mean((first >= 0) & (reached == state))) for state in states}


@analysis(DataKind.SERIES)
def return_times(values: npt.ArrayLike, level: float = 0.0, times: npt.ArrayLike | None = None) -> Array:
    """Return the times between successive returns of a series to a level, by visit or by crossing.

    Args:
        values:
            The series.

        level:
            The level.

        times:
            The times of the values, or `None` for step indices.

    Returns:
        times:
            The time between every return and the next.

    Examples:
        >>> from skcomplex.core.analysis import return_times
        >>> return_times([0, 1, 0, -1, -2, -1, 0]).tolist()
        [2.0, 4.0]
    """
    array = np.asarray(values, dtype=float) - level
    clock = _clock(times, array.size)
    signs = np.sign(array)
    crossing = np.concatenate([[False], signs[1:] * signs[:-1] < 0])
    returns = np.flatnonzero((signs == 0) | crossing)
    return np.diff(clock[returns])


@analysis(DataKind.SERIES)
def laminar_durations(
    values: npt.ArrayLike,
    threshold: float,
    times: npt.ArrayLike | None = None,
) -> Array:
    """Return the durations of the laminar phases of a series, the runs of values within a threshold of 0.

    Args:
        values:
            The series, such as the distance from an invariant manifold of on-off intermittency.

        threshold:
            The largest absolute value of a laminar phase.

        times:
            The times of the values, or `None` for step indices.

    Returns:
        durations:
            The duration of every complete laminar phase, between two bursts.

    Examples:
        >>> from skcomplex.core.analysis import laminar_durations
        >>> laminar_durations([1, 0, 0, 1, 0, 1], 0.5).tolist()
        [2.0, 1.0]
    """
    array = np.abs(np.asarray(values, dtype=float))
    clock = _clock(times, array.size)
    step = float(np.median(np.diff(clock))) if clock.size > 1 else 1.0
    laminar = np.concatenate([[False], array <= threshold, [False]]).astype(int)
    starts, ends = np.flatnonzero(np.diff(laminar) == 1), np.flatnonzero(np.diff(laminar) == -1)
    complete = (starts > 0) & (ends < array.size)
    return clock[ends[complete] - 1] - clock[starts[complete]] + step
