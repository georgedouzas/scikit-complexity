"""Keep the history that delayed laws read, locate and apply the events of hybrid systems, and step through both."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import bisect
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import brentq

from ...core.errors import SolverError
from ...core.types import Vector
from ..problems import CompiledEvent, EventRecord, Problem, Record

type Interpolant = Callable[[float], Vector]
type Bounded = Callable[[float, Vector, Vector], Vector]
type Take = Callable[[float, Vector, float, float], tuple[Vector, Interpolant]]
SEPARATION = 1e-10
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

    def __init__(self, problem: Problem, start: float, state: Vector, *, extrapolate: bool = False) -> None:
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

    def value(self, time: float, indices: Any) -> Vector:  # noqa: ANN401
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
    def _with_delays(problem: Problem, history: 'History | None', t: float, x: Vector, p: Vector) -> Vector:
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
    def _open_history(problem: Problem, history: 'History | None', t: float, x: Vector) -> 'History | None':
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


class EventLocator:
    """Locates events between two times of a step, applies resets, and records them.

    Args:
        problem:
            The compiled problem.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.records: list[Record] = []
        self.stopped_at: float | None = None
        self._last: dict[str, float] = {}

    def crossing(
        self,
        t0: float,
        t1: float,
        x0: Vector,
        x1: Vector,
        *,
        p: Vector,
        interpolant: Interpolant,
        include_hidden: bool = True,
    ) -> tuple[float, int] | None:
        """Return the earliest event crossing in a step, with its time and the index of its event.

        Args:
            t0:
                The start of the step.

            t1:
                The end of the step.

            x0:
                The state at the start.

            x1:
                The state at the end.

            p:
                The parameter vector.

            interpolant:
                The state as a function of time within the step.

            include_hidden:
                Whether to locate hidden events, which only restart adaptive integrators at switches of the drift.
                Fixed-step integrators step over those switches, so they leave them out.

        Returns:
            crossing:
                The time and the event index, or `None` when no guard crosses zero in its direction.
        """
        earliest: tuple[float, int] | None = None
        for index, event in enumerate(self.problem.events):
            if event.hidden and not include_hidden:
                continue
            g0, g1 = event.guard(t0, x0, p), event.guard(t1, x1, p)
            crossed = (
                (event.direction >= 0 and g0 < 0 < g1)
                or (event.direction <= 0 and g0 > 0 > g1)
                or (event.direction >= 0 and g0 < 0 == g1)
                or (event.direction <= 0 and g0 > 0 == g1)
            )
            if not crossed:
                continue
            time = t1 if g1 == 0 else self._past_root(event, g0, t0, t1, p=p, interpolant=interpolant)
            if earliest is None or time < earliest[0]:
                earliest = (time, index)
        return earliest

    @staticmethod
    def _past_root(
        event: CompiledEvent,
        g0: float,
        t0: float,
        t1: float,
        *,
        p: Vector,
        interpolant: Interpolant,
    ) -> float:
        """Return the root of a guard in a step, moved just past it so the guard has already changed sign there."""

        def guard(s: float) -> float:
            return event.guard(s, interpolant(s), p)

        time = brentq(guard, t0, t1, xtol=1e-13, rtol=1e-12)
        nudge = 1e-15 * max(1.0, abs(time))
        while time < t1 and np.sign(guard(time)) == np.sign(g0):
            time, nudge = min(t1, time + nudge), 2 * nudge
        return time

    def apply(self, time: float, index: int, state: Vector, p: Vector) -> Vector:
        """Apply an event's reset at its time, record it, and stop the run when it is terminal.

        Args:
            time:
                The time of the crossing.

            index:
                The index of the event.

            state:
                The state at the crossing.

            p:
                The parameter vector.

        Returns:
            state:
                The state after the reset.

        Raises:
            SolverError:
                The event fires again within the minimum separation, which signals chattering.
        """
        event = self.problem.events[index]
        if event.hidden:
            return state.copy()
        last = self._last.get(event.label)
        if last is not None and time - last < SEPARATION * max(1.0, abs(time)):
            error_msg = (
                f'Event {event.label} fired again at t = {time} right after its previous firing, which signals '
                'chattering. Change the reset so that the guard leaves zero, or end the run with a terminal event.'
            )
            raise SolverError(error_msg)
        self._last[event.label] = time
        after = event.reset(time, state, p) if event.reset is not None else state.copy()
        self.records.append(EventRecord(event.label, time, state.copy(), after.copy()))
        if event.terminal:
            self.stopped_at = time
        return after

    @staticmethod
    def _find_bounds(problem: Problem, t: float, until: float) -> list[float]:
        """Return the breakpoints between two times followed by the end time.

        Args:
            problem:
                The compiled problem.

            t:
                The start time.

            until:
                The end time.

        Returns:
            bounds:
                The times a step must not cross, in increasing order.
        """
        return [*problem.breakpoints.between(t, until), until]

    @staticmethod
    def _find_left_limit(bound: float) -> float:
        """Return the last time before a bound, where inputs that jump at the bound still take their earlier value.

        Args:
            bound:
                The bound.

        Returns:
            time:
                The largest float below the bound.
        """
        return float(np.nextafter(bound, -math.inf))

    @staticmethod
    def _find_smallest_delay(problem: Problem) -> float:
        """Return the smallest delay of a problem, or infinity without delays.

        Args:
            problem:
                The compiled problem.

        Returns:
            delay:
                The smallest delay.
        """
        return min((delay.minimum for delay in problem.delays), default=math.inf)


