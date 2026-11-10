"""Define the breakpoints spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import dataclass, field

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _BreakpointsSpec(BaseSpec):
    """Times where time-dependent inputs jump, at which solvers restart."""

    fixed: tuple[float, ...] = ()
    periodic: tuple[tuple[float, float], ...] = field(default=())
    unresolved: bool = False

    def get_next_after(self, time: float) -> float | None:
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

    def get_between(self, start: float, end: float) -> list[float]:
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
