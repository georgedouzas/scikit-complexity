"""Define continuous time, in which rates, noise, equations, jumps, and crossings change the states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseTime


class ContinuousTime(BaseTime):
    """Continuous time, in which rates, noise, implicit equations, jumps, and crossings change the states.

    Args:
        label:
            The label of the time in its system.

    Examples:
        >>> from skcomplex.core.times import ContinuousTime
        >>> ContinuousTime().check().prepare().schedule_ is None
        True
    """

    def __init__(self, *, label: str = 'time') -> None:
        super().__init__(label)

    def _prepare(self, context: None = None) -> None:
        """Set no schedule."""
        self.schedule_ = None
