"""Define static time, in which a system holds only constraints and objectives and a run solves it once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ._base import BaseTime


class StaticTime(BaseTime):
    """No time: the system holds only constraints and objectives, and a run solves it once.

    Args:
        label:
            The label of the time in its system.

    Examples:
        >>> from skcomplex.core.spaces import StaticTime
        >>> StaticTime().prepare().schedule_ is None
        True
    """

    def __init__(self, *, label: str = 'time') -> None:
        super().__init__(label)

    def _prepare(self, context: None = None) -> None:
        """Set no schedule."""
        self.schedule_ = None

    def _get_frames(self, until: float) -> npt.NDArray[np.float64]:
        """Return the one frame of the solution."""
        return np.zeros(1)
