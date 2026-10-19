"""Pace simulations against wall time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import time

from sklearn.base import BaseEstimator


class Clock(BaseEstimator):
    """Base class of clocks, which decide how long to wait between frames."""

    def start(self) -> None:
        """Mark the wall time at which the run starts."""
        self.started_at_ = time.perf_counter()
        self.max_lag_ = 0.0

    def wait(self, elapsed: float) -> None:
        """Wait until a frame that is `elapsed` simulated time units after the start is due.

        Args:
            elapsed:
                The simulated time since the start of the run.
        """


class AsFastAsPossible(Clock):
    """Clock that never waits.

    Examples:
        >>> from skcomplex.runtime import AsFastAsPossible
        >>> clock = AsFastAsPossible()
        >>> clock.start()
        >>> clock.wait(10.0)
    """


class RealTime(Clock):
    """Clock that advances simulated time at a fixed speed relative to wall time.

    The largest delay of a frame behind its due time is recorded in `max_lag_`.

    Args:
        speed:
            The number of simulated time units per second of wall time.

    Examples:
        >>> from skcomplex.runtime import RealTime
        >>> clock = RealTime(speed=100.0)
        >>> clock.start()
        >>> clock.wait(1.0)
        >>> clock.max_lag_ < 0.1
        True
    """

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = speed

    def wait(self, elapsed: float) -> None:
        """Sleep until the frame is due, or record the lag when the run is behind.

        Args:
            elapsed:
                The simulated time since the start of the run.
        """
        delay = self.started_at_ + elapsed / self.speed - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
            return
        self.max_lag_ = max(self.max_lag_, -delay)
