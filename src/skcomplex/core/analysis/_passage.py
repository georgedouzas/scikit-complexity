"""Measure first passages, absorption, returns, and committors of paths."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydtmc import MarkovChain

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from ._markov import MarkovAnalysis
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


class PassageAnalysis(BaseAnalysis):
    """Passages of paths to a level or a set of states, or of a Markov chain between two sets, of one kind.

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

    Attributes:
        values_:
            The paths, one row per path, or the series of `'return-times'`.

        times_:
            The common times of the steps.

        matrix_:
            The transition matrix of the matrix kinds.

        passage_times_:
            The time from the start to the first passage of every path, `nan` for paths that never pass, of
            `'first-passage'`.

        hitting_probabilities_:
            The share of paths that hit every target state first, of `'hitting'`.

        return_times_:
            The time between every return and the next, of `'return-times'`.

        committor_:
            The committor of every state, 0 on the origin and 1 on the target, of `'committor'`.

        mean_first_passage_time_:
            The mean first-passage time, of `'mean-first-passage'`.

    Examples:
        >>> from skcomplex.core.analysis import PassageAnalysis
        >>> passage = PassageAnalysis(threshold=2).prepare([[0, 1, 2, 3], [0, 0, 0, 0]]).run()
        >>> passage.passage_times_.tolist()
        [2.0, nan]
        >>> PassageAnalysis('return-times').prepare([0, 1, 0, -1, -2, -1, 0]).run().return_times_.tolist()
        [2.0, 4.0]
        >>> walk = [[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]]
        >>> PassageAnalysis('committor', origin=[0], target=[2]).prepare(walk).run().committor_.tolist()
        [0.0, 0.5, 1.0]
    """

    def __init__(
        self,
        kind: str = 'first-passage',
        variables: Sequence[str] | None = None,
        *,
        threshold: float | None = None,
        target: Sequence[float] | None = None,
        origin: Sequence[float] | None = None,
        direction: str = 'up',
        level: float = 0.0,
        lag: int = 1,
        dt: float = 1.0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.threshold = threshold
        self.target = target
        self.origin = origin
        self.direction = direction
        self.level = level
        self.lag = lag
        self.dt = dt

    def _check(self) -> None:
        """Validate the kind, the direction, the sets each kind needs, and the step."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('direction', self.direction, DIRECTIONS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.kind == 'first-passage' and self.threshold is None and self.target is None:
            error_msg = f'{type(self).__name__} needs a threshold or the target states that end the passage.'
            raise AnalysisError(error_msg)
        if self.kind == 'hitting' and self.target is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs the absorbing target states.'
            raise AnalysisError(error_msg)
        if self.kind in MATRIX_KINDS and (self.origin is None or self.target is None):
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs the origin and the target states.'
            raise AnalysisError(error_msg)
        if self.lag < 1 or self.dt <= 0:
            error_msg = f'{type(self).__name__} has lag={self.lag!r} and dt={self.dt!r}. Use positive values.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the series kind for returns, and the path kind otherwise."""
        return DataKind.SERIES if self.kind == 'return-times' else DataKind.PATH

    def _prepare(self, source: Source) -> None:
        """Read the transition matrix of the matrix kinds, and the paths or the series otherwise."""
        name = self.variables[0] if self.variables else None
        if self.kind in MATRIX_KINDS:
            if isinstance(source, Trajectory):
                states = np.asarray(self._read_path(source, name).states, dtype=np.int64)
                self.matrix_ = MarkovAnalysis._transition_matrix(states, lag=self.lag)
            else:
                self.matrix_ = np.asarray(source, dtype=float)
            return
        if self.kind == 'return-times':
            series = self._read_series(source, self.variables)
            self.values_ = series.values[:, 0]
            self.times_ = series.times if isinstance(source, Trajectory) else series.times * self.dt
            return
        if isinstance(source, Trajectory):
            path = self._read_path(source, name)
            self.values_ = self._rows(path.states)
            self.times_ = np.asarray(path.times, dtype=float)
            return
        self.values_ = self._rows(source)
        self.times_ = np.arange(self.values_.shape[1], dtype=float) * self.dt

    def _get_states(self, states: Sequence[float] | None) -> list[float]:
        """Return a set of states as a list, empty without states."""
        return [] if states is None else list(states)

    def _run_first_passage(self) -> None:
        """Compute the first passage of every path to the threshold or into the target states."""
        threshold = float('nan') if self.threshold is None else self.threshold
        if self.target is not None:
            hits = np.isin(self.values_, np.asarray(self.target, dtype=float))
        elif self.direction == 'up':
            hits = self.values_ >= threshold
        else:
            hits = self.values_ <= threshold
        first = self._first_hits(hits)
        clock = self.times_
        self.passage_times_ = np.where(first >= 0, clock[np.maximum(first, 0)] - clock[0], np.nan)

    def _run_hitting(self) -> None:
        """Compute the share of paths that reach every target state first."""
        target = self._get_states(self.target)
        first = self._first_hits(np.isin(self.values_, np.asarray(target, dtype=float)))
        reached = self.values_[np.arange(self.values_.shape[0]), np.maximum(first, 0)]
        self.hitting_probabilities_ = np.asarray(
            [np.mean((first >= 0) & (reached == state)) for state in target],
            dtype=float,
        )

    def _run_returns(self) -> None:
        """Compute the times between the returns of the series to the level."""
        array = self.values_ - self.level
        signs = np.sign(array)
        crossing = np.concatenate([[False], signs[1:] * signs[:-1] < 0])
        returns = np.flatnonzero((signs == 0) | crossing)
        self.return_times_ = np.diff(self.times_[returns])

    def _get_indices(self, states: Sequence[float] | None) -> list[int]:
        """Return a set of states as the indices of the rows of the matrix."""
        return [int(state) for state in self._get_states(states)]

    def _run_committor(self) -> None:
        """Compute the forward committor from the origin to the target states."""
        chain = MarkovChain(self.matrix_)
        origin, target = self._get_indices(self.origin), self._get_indices(self.target)
        self.committor_ = np.asarray(chain.committor_probabilities('forward', origin, target), dtype=float)

    def _run_mean_first_passage(self) -> None:
        """Compute the mean first-passage time to the target states, from the origin in its stationary weights."""
        times = MarkovChain(self.matrix_).first_passage_times_to('mean', self._get_indices(self.target))
        if times is None:
            error_msg = f'{type(self).__name__} needs a transition matrix with one stationary law, not a reducible one.'
            raise AnalysisError(error_msg)
        origin = self._get_indices(self.origin)
        weights = MarkovAnalysis._stationary_distribution(self.matrix_)[origin]
        self.mean_first_passage_time_ = float(weights @ times[origin] / weights.sum() * self.lag * self.dt)

    def _run(self, inputs: object = None) -> None:
        """Compute the passage of the kind."""
        if self.kind == 'first-passage':
            self._run_first_passage()
        elif self.kind == 'hitting':
            self._run_hitting()
        elif self.kind == 'return-times':
            self._run_returns()
        elif self.kind == 'committor':
            self._run_committor()
        else:
            self._run_mean_first_passage()

    @staticmethod
    def _rows(paths: npt.ArrayLike) -> Array:
        """Return paths as one row per path."""
        array = np.asarray(paths, dtype=float)
        return array.reshape(1, -1) if array.ndim == 1 else array

    @staticmethod
    def _first_hits(hits: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
        """Return the first step at which every row hits, or -1 for rows that never hit."""
        first = hits.argmax(axis=1)
        return np.where(hits.any(axis=1), first, -1)
