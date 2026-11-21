"""Run simulations as fast as possible."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseClock


class AsFastAsPossibleClock(BaseClock):
    """A clock that never waits, so a run goes as fast as it can.

    It hands each frame on as soon as the solver has computed it, so `max_lag_` stays 0. It is the clock of every run
    that names none, such as a batch of runs of an epidemic, a sweep of the rates of a reaction, or the training of a
    network.

    Read more in the user guide, under Observe.

    Attributes:
        started_:
            The wall time at which the run started, in seconds of `time.perf_counter`.

        max_lag_:
            The largest delay of a frame behind its due time, always 0.

    See Also:
        RealTimeClock: Runs at a fixed speed against wall time.

    Examples:
        >>> from skcomplex.core.engine.runtime import AsFastAsPossibleClock
        >>> clock = AsFastAsPossibleClock().prepare().run(10.0)
        >>> clock.max_lag_
        0.0
    """
