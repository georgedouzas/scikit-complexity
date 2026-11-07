"""Measure first passages, absorption, returns, and committors of paths."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydtmc import MarkovChain

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import _check_choice, _check_settings, _read_path, _read_series
from ._markov import _stationary_distribution, _transition_matrix
from .types import Source

KINDS = ('first-passage', 'hitting', 'return-times', 'committor', 'mean-first-passage')
DIRECTIONS = ('up', 'down')
SETTINGS = {
    'first-passage': {'threshold', 'target', 'direction', 'dt'},
    'hitting': {'target'},
    'return-times': {'level', 'dt'},
    'committor': {'origin', 'target', 'lag'},
    'mean-first-passage': {'origin', 'target', 'lag', 'dt'},
}
"""Settings every kind reads, besides `kind` and `variables`."""

MATRIX_KINDS = ('committor', 'mean-first-passage')


@dataclass(frozen=True)
class PassageResult:
    """Passages of paths to a level or a set of states, or of a Markov chain between two sets.

    Args:
        values:
            The paths, one row per path, or the series of `'return-times'`.

        times:
            The common times of the steps.

        matrix:
            The transition matrix of the matrix kinds.

        passage_times:
            The time from the start to the first passage of every path, `nan` for paths that never pass, of
            `'first-passage'`.

        hitting_probabilities:
            The share of paths that hit every target state first, of `'hitting'`.

        return_times:
            The time between every return and the next, of `'return-times'`.

        committor:
            The committor of every state, 0 on the origin and 1 on the target, of `'committor'`.

        mean_first_passage_time:
            The mean first-passage time, of `'mean-first-passage'`.
    """

    values: Array | None = None
    times: Array | None = None
    matrix: Array | None = None
    passage_times: Array | None = None
    hitting_probabilities: Array | None = None
    return_times: Array | None = None
    committor: Array | None = None
    mean_first_passage_time: float | None = None


def passage(
    source: Source,
    *,
    kind: str = 'first-passage',
    variables: Sequence[str] | None = None,
    threshold: float | None = None,
    target: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    direction: str = 'up',
    level: float = 0.0,
    lag: int = 1,
    dt: float = 1.0,
) -> PassageResult:
    """Return the passages of paths to a level or a set of states, or of a Markov chain between two sets, of one kind.

    The kinds read paths, one row per path on common times, or a transition matrix:

    - `'first-passage'`: the first time every path reaches `threshold`, from below for `direction='up'` and from
      above for `'down'`, or enters the `target` states.
    - `'hitting'`: the share of paths that reach each of the absorbing `target` states before the others.
    - `'return-times'`: the times between successive returns of a series to `level`, by visit or by crossing.
    - `'committor'`: the forward committor of a transition matrix, the probability that the chain reaches the
      `target` states before the `origin` states.
    - `'mean-first-passage'`: the mean first-passage time of a transition matrix at `lag` from the `origin` states,
      started in their stationary weights, to the `target` states.

    The matrix kinds estimate the transition matrix at `lag` of the jump path of a trajectory.

    Args:
        source:
            The trajectory, an array of paths, one row per path, a series for `'return-times'`, or a transition
            matrix for the matrix kinds.

        kind:
            The passage to measure, one of the kinds above.

        variables:
            The discrete state of a trajectory to read, or the series for `'return-times'`, or `None` for the first.

        threshold:
            The level a first passage reaches.

        target:
            The states that end a passage, the absorbing states, or the target set of the chain.

        origin:
            The origin set of the chain.

        direction:
            `'up'` to reach values at or above the threshold, `'down'` at or below it.

        level:
            The level of the returns.

        lag:
            The lag of the transition matrix in steps.

        dt:
            The time of one step of an array, while a trajectory gives its own times.

    Returns:
        result:
            The paths, the series, or the matrix read, and the passages of the kind.

    Raises:
        AnalysisError:
            A choice is unknown, a setting the kind does not read is changed, the sets the kind needs are missing, a
            setting is not positive, or the matrix is reducible.

    Examples:
        >>> from skcomplex.core.analysis import passage
        >>> passage([[0, 1, 2, 3], [0, 0, 0, 0]], threshold=2).passage_times.tolist()
        [2.0, nan]
        >>> passage([0, 1, 0, -1, -2, -1, 0], kind='return-times').return_times.tolist()
        [2.0, 4.0]
        >>> walk = [[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]]
        >>> passage(walk, kind='committor', origin=[0], target=[2]).committor.tolist()
        [0.0, 0.5, 1.0]
    """
    _check_choice('passage', 'kind', kind, KINDS)
    _check_choice('passage', 'direction', direction, DIRECTIONS)
    _check_settings(
        passage,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        threshold=threshold,
        target=target,
        origin=origin,
        direction=direction,
        level=level,
        lag=lag,
        dt=dt,
    )
    _check(kind, threshold=threshold, target=target, origin=origin, lag=lag, dt=dt)
    name = variables[0] if variables else None
    if kind in MATRIX_KINDS:
        if isinstance(source, Trajectory):
            states = np.asarray(_read_path(source, name).states, dtype=np.int64)
            matrix = _transition_matrix(states, lag=lag)
        else:
            matrix = np.asarray(source, dtype=float)
        if kind == 'committor':
            chain = MarkovChain(matrix)
            committor = chain.committor_probabilities('forward', _get_indices(origin), _get_indices(target))
            return PassageResult(matrix=matrix, committor=np.asarray(committor, dtype=float))
        time = _mean_first_passage(matrix, origin, target, lag, dt)
        return PassageResult(matrix=matrix, mean_first_passage_time=time)
    if kind == 'return-times':
        series = _read_series(source, variables)
        times = series.times if isinstance(source, Trajectory) else series.times * dt
        return _returns(series.values[:, 0], times, level)
    if isinstance(source, Trajectory):
        path = _read_path(source, name)
        values = _rows(path.states)
        times = np.asarray(path.times, dtype=float)
    else:
        values = _rows(source)
        times = np.arange(values.shape[1], dtype=float) * dt
    if kind == 'first-passage':
        return _first_passage(values, times, threshold, target, direction)
    targets = _get_states(target)
    first = _first_hits(np.isin(values, np.asarray(targets, dtype=float)))
    reached = values[np.arange(values.shape[0]), np.maximum(first, 0)]
    probabilities = np.asarray([np.mean((first >= 0) & (reached == state)) for state in targets], dtype=float)
    return PassageResult(values, times, hitting_probabilities=probabilities)


def _check(
    kind: str,
    *,
    threshold: float | None,
    target: Sequence[float] | None,
    origin: Sequence[float] | None,
    lag: int,
    dt: float,
) -> None:
    """Raise an error for a set a kind needs that is missing, or a lag or a step that is not positive."""
    if kind == 'first-passage' and threshold is None and target is None:
        error_msg = 'passage needs a threshold or the target states that end the passage.'
        raise AnalysisError(error_msg)
    if kind == 'hitting' and target is None:
        error_msg = f'passage of kind {kind!r} needs the absorbing target states.'
        raise AnalysisError(error_msg)
    if kind in MATRIX_KINDS and (origin is None or target is None):
        error_msg = f'passage of kind {kind!r} needs the origin and the target states.'
        raise AnalysisError(error_msg)
    if lag < 1 or dt <= 0:
        error_msg = f'passage has lag={lag!r} and dt={dt!r}. Use positive values.'
        raise AnalysisError(error_msg)


def _returns(values: Array, times: Array, level: float) -> PassageResult:
    """Return the times between the returns of the series to the level."""
    signs = np.sign(values - level)
    crossing = np.concatenate([[False], signs[1:] * signs[:-1] < 0])
    returns = np.flatnonzero((signs == 0) | crossing)
    return PassageResult(values, times, return_times=np.diff(times[returns]))


def _first_passage(
    values: Array,
    times: Array,
    threshold: float | None,
    target: Sequence[float] | None,
    direction: str,
) -> PassageResult:
    """Return the first passage of every path to the threshold or into the target states."""
    level = float('nan') if threshold is None else threshold
    if target is not None:
        hits = np.isin(values, np.asarray(target, dtype=float))
    elif direction == 'up':
        hits = values >= level
    else:
        hits = values <= level
    first = _first_hits(hits)
    passage_times = np.where(first >= 0, times[np.maximum(first, 0)] - times[0], np.nan)
    return PassageResult(values, times, passage_times=passage_times)


def _get_states(states: Sequence[float] | None) -> list[float]:
    """Return a set of states as a list, empty without states."""
    return [] if states is None else list(states)


def _get_indices(states: Sequence[float] | None) -> list[int]:
    """Return a set of states as the indices of the rows of the matrix."""
    return [int(state) for state in _get_states(states)]


def _mean_first_passage(
    matrix: Array,
    origin: Sequence[float] | None,
    target: Sequence[float] | None,
    lag: int,
    dt: float,
) -> float:
    """Return the mean first-passage time to the target states, from the origin in its stationary weights."""
    times = MarkovChain(matrix).first_passage_times_to('mean', _get_indices(target))
    if times is None:
        error_msg = 'passage needs a transition matrix with one stationary law, not a reducible one.'
        raise AnalysisError(error_msg)
    start = _get_indices(origin)
    weights = _stationary_distribution(matrix)[start]
    return float(weights @ times[start] / weights.sum() * lag * dt)


def _rows(paths: npt.ArrayLike) -> Array:
    """Return paths as one row per path."""
    array = np.asarray(paths, dtype=float)
    return array.reshape(1, -1) if array.ndim == 1 else array


def _first_hits(hits: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    """Return the first step at which every row hits, or -1 for rows that never hit."""
    first = hits.argmax(axis=1)
    return np.where(hits.any(axis=1), first, -1)
