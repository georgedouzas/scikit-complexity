"""Pace simulations at a fixed speed against wall time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import time

from ...errors import ValidationError
from ._base import BaseClock


class RealTimeClock(BaseClock):
    """A clock that runs simulated time at a fixed speed against wall time.

    It waits before each frame until its time is due, so `RealTimeClock(speed=100.0)` shows 100 time units per second of
    wall time, and it records in `max_lag_` how far a frame fell behind when the solver was slower. It paces a live view
    of a pendulum, a demo of a traffic jam, and a game against a model of a market.

    Read more in the user guide, under Observe.

    Args:
        speed:
            The number of simulated time units per second of wall time.

    Attributes:
        started_:
            The wall time at which the run started, in seconds of `time.perf_counter`.

        max_lag_:
            The largest delay of a frame behind its due time, in seconds.

    See Also:
        AsFastAsPossibleClock: Never waits.

    Examples:
        >>> from skcomplex.core.engine.runtime import RealTimeClock
        >>> clock = RealTimeClock(speed=100.0).prepare().run(1.0)
        >>> clock.max_lag_ < 0.1
        True
    """

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = speed

    def _prepare(self, context: None) -> None:
        """Raise an error unless the speed is a positive number."""
        super()._prepare(context)
        if not isinstance(self.speed, int | float) or isinstance(self.speed, bool) or not self.speed > 0:
            error_msg = f'speed = {self.speed!r} must be a positive number.'
            raise ValidationError(error_msg)

    def _run(self, inputs: float | None) -> None:
        """Sleep until the frame is due, or record the lag when the run is behind.

        Args:
            inputs:
                The simulated time since the start of the run.

        Raises:
            ValidationError:
                No simulated time is passed.
        """
        if inputs is None:
            error_msg = 'A clock runs on the simulated time since the start of the run. Pass it to `run`.'
            raise ValidationError(error_msg)
        delay = self.started_ + inputs / self.speed - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
            return
        self.max_lag_ = max(self.max_lag_, -delay)
