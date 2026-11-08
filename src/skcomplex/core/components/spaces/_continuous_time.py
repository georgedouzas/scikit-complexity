"""Define continuous time.

In continuous time, rates, noise, equations, jumps, and crossings change the states.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ._base import BaseTime


class ContinuousTime(BaseTime):
    """Continuous time, in which rates and other continuous laws change the states.

    The laws are rates, noise, implicit equations, jumps, and crossings.

    Args:
        label:
            The label of the time in its system.

    Examples:
        >>> from skcomplex.core.components.spaces import ContinuousTime
        >>> ContinuousTime().prepare().schedule_ is None
        True
    """

    time_tag: ClassVar[str] = 'continuous'

    def __init__(self, *, label: str = 'time') -> None:
        super().__init__(label)

    def _prepare(self, context: None = None) -> None:
        """Set no schedule."""
        self.schedule_ = None
