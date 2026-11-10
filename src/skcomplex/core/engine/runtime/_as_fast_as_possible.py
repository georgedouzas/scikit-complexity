"""Run simulations as fast as possible."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseClock


class AsFastAsPossibleClock(BaseClock):
    """A clock that never waits, so a run goes as fast as it can.

    Examples:
        >>> from skcomplex.core.engine.runtime import AsFastAsPossibleClock
        >>> clock = AsFastAsPossibleClock().prepare().run(10.0)
        >>> clock.max_lag_
        0.0
    """
