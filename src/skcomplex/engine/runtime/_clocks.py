"""Pace simulations against wall time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import time

from ...core.base import BaseEstimator


class BaseClock(BaseEstimator):
    """Base class of clocks, which decide how long to wait between frames.

    A clock is prepared at the start of a run with `prepare()`, which marks the wall time, and `run(elapsed)` waits
    until a frame that is `elapsed` simulated time units after the start is due. Subclasses implement `_run`.

    Attributes:
        started_:
            The wall time at which the run started, in seconds of `time.perf_counter`.

        max_lag_:
            The largest delay of a frame behind its due time, in seconds.
    """

    def _prepare(self) -> None:
        """Mark the wall time at which the run starts."""
        self.started_ = time.perf_counter()
        self.max_lag_ = 0.0

    def _run(self, elapsed: float) -> None:
        """Wait until a frame that is `elapsed` simulated time units after the start is due; never wait by default.

        Args:
            elapsed:
                The simulated time since the start of the run.
        """


class AsFastAsPossibleClock(BaseClock):
    """Clock that never waits.

    Examples:
        >>> from skcomplex.engine.runtime import AsFastAsPossibleClock
        >>> clock = AsFastAsPossibleClock().prepare()
        >>> clock.run(10.0)
    """


class RealTimeClock(BaseClock):
    """Clock that advances simulated time at a fixed speed relative to wall time.

    The largest delay of a frame behind its due time is recorded in `max_lag_`.

    Args:
        speed:
            The number of simulated time units per second of wall time.

    Examples:
        >>> from skcomplex.engine.runtime import RealTimeClock
        >>> clock = RealTimeClock(speed=100.0).prepare()
        >>> clock.run(1.0)
        >>> clock.max_lag_ < 0.1
        True
    """

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = speed

    def _run(self, elapsed: float) -> None:
        """Sleep until the frame is due, or record the lag when the run is behind.

        Args:
            elapsed:
                The simulated time since the start of the run.
        """
        delay = self.started_ + elapsed / self.speed - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
            return
        self.max_lag_ = max(self.max_lag_, -delay)
