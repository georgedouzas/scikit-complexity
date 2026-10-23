"""Describe the hybrid parts of a problem: delayed states, discrete events, and the breakpoints of time inputs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..base import Expression
from ._arrays import Vector


@dataclass(frozen=True)
class Delay:
    """Delayed state read by laws through a hidden parameter that solvers fill from the run's history.

    Args:
        parameters:
            The flat indices of the hidden parameter, one per member.

        states:
            The flat indices of the delayed state component, aligned with `parameters`.

        tau:
            The function `tau(t, x, p)` that returns the delay.

        minimum:
            The smallest value of the delay, which bounds the steps of the method of steps.

        symbol:
            The symbol that replaces the delayed value in the equations.

        state:
            The symbol of the delayed state.

        delay:
            The symbolic delay.
    """

    parameters: npt.NDArray[np.intp]
    states: npt.NDArray[np.intp]
    tau: Callable[[float, Vector, Vector], float]
    minimum: float
    symbol: Expression
    state: Expression
    delay: Expression


@dataclass(frozen=True)
class CompiledEvent:
    """Event compiled into a guard function and a reset function of the state vector.

    Args:
        label:
            The name of the event.

        guard:
            The function `g(t, x, p)` whose crossing of zero triggers the event.

        reset:
            The function `r(t, x, p)` that returns the state vector after the event, or `None` to keep it.

        direction:
            `1`, `-1`, or `0` for crossings upwards, downwards, or both.

        terminal:
            Whether the event ends the run.

        hidden:
            Whether the event only marks a switch of a law at a condition on states, so integrators restart at the
            crossing, without a reset and without a record in the trajectory.
    """

    label: str
    guard: Callable[[float, Vector, Vector], float]
    reset: Callable[[float, Vector, Vector], Vector] | None
    direction: int
    terminal: bool
    hidden: bool = False


@dataclass(frozen=True)
class EventRecord:
    """Event that happened during a run.

    Args:
        label:
            The name of the event.

        time:
            The time of the crossing.

        before:
            The state vector at the crossing, before the reset.

        after:
            The state vector after the reset.
    """

    label: str
    time: float
    before: Vector
    after: Vector


@dataclass(frozen=True)
class Breakpoints:
    """Times where time-dependent inputs jump, at which solvers restart.

    Args:
        fixed:
            Single times.

        periodic:
            Pairs of an origin and a period, each giving the times `origin + k period`.

        unresolved:
            Whether a condition on time has jumps at times that depend on parameters, so solvers restart at every
            frame instead.
    """

    fixed: tuple[float, ...] = ()
    periodic: tuple[tuple[float, float], ...] = field(default=())
    unresolved: bool = False

    def next_after(self, time: float) -> float | None:
        """Return the first breakpoint strictly after a time, or `None` when none follows.

        Args:
            time:
                The time.

        Returns:
            breakpoint:
                The first later breakpoint.
        """
        candidates = [item for item in self.fixed if item > time]
        for origin, period in self.periodic:
            following = origin + (math.floor((time - origin) / period) + 1) * period
            while following <= time:
                following += period
            candidates.append(following)
        return min(candidates, default=None)

    def between(self, start: float, end: float) -> list[float]:
        """Return the breakpoints strictly between two times, in order.

        Args:
            start:
                The first time.

            end:
                The last time.

        Returns:
            times:
                The breakpoints.
        """
        times = {time for time in self.fixed if start < time < end}
        for origin, period in self.periodic:
            first = math.floor((start - origin) / period) + 1
            last = math.ceil((end - origin) / period)
            times.update(origin + k * period for k in range(first, last) if start < origin + k * period < end)
        return sorted(times)


