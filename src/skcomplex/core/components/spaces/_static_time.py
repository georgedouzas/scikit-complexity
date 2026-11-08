"""Define static time.

In static time a system holds only constraints and objectives, and a run solves it once.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ._base import BaseTime


class StaticTime(BaseTime):
    """No time.

    The system holds only constraints and objectives, and a run solves it once.

    Args:
        label:
            The label of the time in its system.

    Examples:
        >>> from skcomplex.core.components.spaces import StaticTime
        >>> StaticTime().prepare().schedule_ is None
        True
    """

    time_tag: ClassVar[str] = 'static'

    def __init__(self, *, label: str = 'time') -> None:
        super().__init__(label)

    def _prepare(self, context: None = None) -> None:
        """Set no schedule."""
        super()._prepare(context)
        self.schedule_ = None

    def _get_frames(self, until: float) -> npt.NDArray[np.float64]:
        """Return the one frame of the solution."""
        return np.zeros(1)
