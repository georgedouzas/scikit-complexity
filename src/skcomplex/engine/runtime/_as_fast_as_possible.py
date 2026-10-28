"""Run simulations as fast as possible."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseClock


class AsFastAsPossibleClock(BaseClock):
    """Clock that never waits.

    Examples:
        >>> from skcomplex.engine.runtime import AsFastAsPossibleClock
        >>> clock = AsFastAsPossibleClock().check().prepare().run(10.0)
        >>> clock.max_lag_
        0.0
    """
