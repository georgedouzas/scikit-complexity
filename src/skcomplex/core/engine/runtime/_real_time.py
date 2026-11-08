"""Pace simulations at a fixed speed against wall time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import time

from ...errors import ValidationError
from ._base import BaseClock


class RealTimeClock(BaseClock):
    """Clock that advances simulated time at a fixed speed relative to wall time.

    The largest delay of a frame behind its due time is recorded in `max_lag_`.

    Args:
        speed:
            The number of simulated time units per second of wall time.

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
