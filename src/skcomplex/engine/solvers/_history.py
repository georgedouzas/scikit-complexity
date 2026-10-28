"""Past states of a run, as interpolants over time segments, and the declared history before the start."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import bisect
from typing import Any

import numpy as np

from ...errors import SolverError
from ...types import Array, Interpolant
from ..problems import Problem

_RESTART = 1e-12


class History:
    """Past states of a run, as interpolants over time segments, and the declared history before the start.

    Args:
        problem:
            The compiled problem.

        start:
            The time the run starts.

        state:
            The state vector at the start.

        extrapolate:
            Whether a time after the last recorded segment reads the extension of that segment, for adaptive solvers
            whose steps outgrow the shortest delay, instead of raising an error.
    """

    def __init__(self, problem: Problem, start: float, state: Array, *, extrapolate: bool = False) -> None:
        self.start = start
        self.extrapolate = extrapolate
        self.initial = state.copy()
        self.declared: dict[int, Any] = {}
        for name, value in problem.history.items():
            entry = problem.states.entry(name)
            for offset in range(entry.size):
                self.declared[entry.start + offset] = (
                    (value, offset)
                    if callable(value)
                    else np.ravel(np.asarray(value, dtype=float))[offset % np.size(value)]
                )
        self.starts: list[float] = []
        self.segments: list[tuple[float, float, Interpolant]] = []
        self.span = max((delay.minimum for delay in problem.delays), default=0.0)

    def add(self, first: float, last: float, interpolant: Interpolant) -> None:
        """Add the states between two times, and drop segments older than any delay can reach.

        Args:
            first:
                The start of the segment.

            last:
                The end of the segment.

            interpolant:
                The function that returns the state vector at a time of the segment.
        """
        self.starts.append(first)
        self.segments.append((first, last, interpolant))

    def truncate(self, time: float) -> None:
        """End the last segment at a time, such as the time of an event that changed the state.

        Args:
            time:
                The new end of the last segment.
        """
        if self.segments:
            first, _, interpolant = self.segments[-1]
            self.segments[-1] = (first, time, interpolant)

    def _declared(self, index: int, time: float) -> float:
        """Return a component of the declared history at a time before the start."""
        value = self.declared[index]
        if isinstance(value, tuple):
            function, offset = value
            values = np.ravel(np.asarray(function(time), dtype=float))
            return float(values[offset % values.size])
        return float(value)

    def value(self, time: float, indices: Any) -> Array:  # noqa: ANN401
        """Return state components at a past time, from the run or from the declared history before the start.

        Args:
            time:
                The past time.

            indices:
                The flat state indices.

        Returns:
            values:
                The values of the components.

        Raises:
            SolverError:
                The time is after the last recorded segment, and the history does not extrapolate.
        """
        if time <= self.start:
            return np.asarray(
                [
                    self._declared(index, time) if index in self.declared else self.initial[index]
                    for index in map(int, indices)
                ],
                dtype=float,
            )
        position = bisect.bisect_right(self.starts, time) - 1
        if position < 0:
            return self.initial[indices]
        _, last, interpolant = self.segments[position]
        if self.extrapolate:
            return np.asarray(interpolant(time), dtype=float)[indices]
        if time > last + 1e-9 * max(1.0, abs(last)):
            error_msg = (
                f'A delay reads the state at t = {time}, after the last integrated time {last}. Keep solver steps '
                'below the smallest delay.'
            )
            raise SolverError(error_msg)
        return np.asarray(interpolant(min(time, last)), dtype=float)[indices]

    @staticmethod
    def _with_delays(problem: Problem, history: 'History | None', t: float, x: Array, p: Array) -> Array:
        """Return the parameter vector with every delayed state read from the history.

        Args:
            problem:
                The compiled problem.

            history:
                The history of the run, or `None` without delays.

            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            parameters:
                The parameters with the hidden parameters of delays filled.
        """
        if history is None or not problem.delays:
            return p
        filled = p.copy()
        for delay in problem.delays:
            filled[delay.parameters] = history.value(t - max(delay.tau(t, x, p), 0.0), delay.states)
        return filled

    @staticmethod
    def _open_history(problem: Problem, history: 'History | None', t: float, x: Array) -> 'History | None':
        """Return the history of a run, started at a time and state when the problem has delays and none is kept yet.

        Args:
            problem:
                The compiled problem.

            history:
                The history kept so far, or `None`.

            t:
                The start time.

            x:
                The state at the start.

        Returns:
            history:
                The history, or `None` without delays.
        """
        if problem.delays and history is None:
            return History(problem, t, x)
        return history
