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
RESTART = 1e-12


def find_bounds(problem: Problem, t: float, until: float) -> list[float]:
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


def find_left_limit(bound: float) -> float:
    """Return the last time before a bound, where inputs that jump at the bound still take their earlier value.

    Args:
        bound:
            The bound.

    Returns:
        time:
            The largest float below the bound.
    """
    return float(np.nextafter(bound, -math.inf))


def find_smallest_delay(problem: Problem) -> float:
    """Return the smallest delay of a problem, or infinity without delays.

    Args:
        problem:
            The compiled problem.

    Returns:
        delay:
            The smallest delay.
    """
    return min((delay.minimum for delay in problem.delays), default=math.inf)


def is_hybrid(problem: Problem, t: float, until: float) -> bool:
    """Check whether a problem needs the hybrid path between two times: delays, events, or breakpoints up to the end.

    Args:
        problem:
            The compiled problem.

        t:
            The start time.

        until:
            The end time.

    Returns:
        hybrid:
            Whether the problem has delays, events, or breakpoints up to and at the end.
    """
    return bool(problem.delays or problem.events or problem.breakpoints.between(t, math.nextafter(until, math.inf)))


def reject_hybrid(problem: Problem, solver: str) -> None:
    """Raise an error when a solver without a hybrid path receives delays or events.

    Args:
        problem:
            The compiled problem.

        solver:
            The name of the solver.

    Raises:
        SolverError:
            The problem has delays or events.
    """
    if problem.delays or problem.events:
        error_msg = (
            f'{solver} does not integrate delays or events. Use ScipyODESolver or RK4Solver for ODEs and '
            'EulerMaruyamaSolver, MilsteinSolver, or HeunSolver for SDEs.'
        )
        raise SolverError(error_msg)


def hermite(t0: float, x0: Vector, f0: Vector, *, t1: float, x1: Vector, f1: Vector) -> Interpolant:
    """Return the cubic Hermite interpolant between two states with their derivatives.

    Args:
        t0:
            The first time.

        x0:
            The first state.

        f0:
            The derivative at the first time.

        t1:
            The second time.

        x1:
            The second state.

        f1:
            The derivative at the second time.

    Returns:
        interpolant:
            The function of time.
    """
    span = t1 - t0

    def evaluate(time: float) -> Vector:
        s = (time - t0) / span if span else 0.0
        return (
            (2 * s**3 - 3 * s**2 + 1) * x0
            + (s**3 - 2 * s**2 + s) * span * f0
            + (-2 * s**3 + 3 * s**2) * x1
            + (s**3 - s**2) * span * f1
        )

    return evaluate


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


def with_delays(problem: Problem, history: History | None, t: float, x: Vector, p: Vector) -> Vector:
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


def open_history(problem: Problem, history: History | None, t: float, x: Vector) -> History | None:
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


def advance_fixed_steps(
    t: float,
    x: Vector,
    p: Vector,
    until: float,
    *,
    dt: float,
    locator: EventLocator,
    history: History | None,
    bounded: Bounded,
    take: Take,
) -> Vector:
    """Integrate with fixed steps that stop at breakpoints, record every step for delays, and locate events in it.

    Args:
        t:
            The current time.

        x:
            The current state vector.

        p:
            The parameter vector.

        until:
            The time to advance to.

        dt:
            The largest step.

        locator:
            The event locator of the run.

        history:
            The history that delayed laws read, or `None` without delays.

        bounded:
            The boundary laws `bounded(t, x, p)`.

        take:
            The step `take(t, x, h, ceiling)` of a scheme, which returns the state after a step of length `h` and its
            interpolant over the step, and reads inputs no later than `ceiling`.

    Returns:
        state:
            The state vector at time `until`, or at a terminal event.

    Raises:
        SolverError:
            The step exceeds the smallest delay.
    """
    problem = locator.problem
    if dt > find_smallest_delay(problem):
        error_msg = f'dt = {dt} exceeds the smallest delay {find_smallest_delay(problem)}. Take a smaller step.'
        raise SolverError(error_msg)
    state, time = x.copy(), t
    for bound in find_bounds(problem, t, until):
        ceiling = find_left_limit(bound)
        while time < bound - RESTART * max(1.0, abs(bound)):
            step = min(dt, bound - time)
            following, interpolant = take(time, state, step, ceiling)
            if history is not None:
                history.add(time, time + step, interpolant)
            hit = (
                locator.crossing(
                    time,
                    time + step,
                    state,
                    following,
                    p=p,
                    interpolant=interpolant,
                    include_hidden=False,
                )
                if problem.events
                else None
            )
            if hit is not None:
                time, index = hit
                if history is not None:
                    history.truncate(time)
                state = bounded(time, locator.apply(time, index, interpolant(time), p), p)
                if locator.stopped_at is not None:
                    return state
                continue
            state, time = bounded(time + step, following, p), time + step
        time = bound
    return state
