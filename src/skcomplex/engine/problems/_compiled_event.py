"""Event compiled into a guard function and a reset function of the state vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...core.types import Vector


@dataclass(frozen=True)
class _Delay:
    """Delayed state read by laws through a hidden parameter that solvers fill from the run's history."""

    parameters: npt.NDArray[np.intp]
    states: npt.NDArray[np.intp]
    tau: Callable[[float, Vector, Vector], float]
    minimum: float
    symbol: Expression
    state: Expression
    delay: Expression


@dataclass(frozen=True)
class _Breakpoints:
    """Times where time-dependent inputs jump, at which solvers restart."""

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
