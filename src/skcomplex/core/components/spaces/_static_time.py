"""Define static time, in which a run solves constraints and objectives once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ._base import BaseTime


class StaticTime(BaseTime):
    """A time with no change, in which a run solves constraints and objectives once.

    The system holds only algebraic equations, constraints, and objectives, and a run solves them once, with no frames.

    Unlike `ContinuousTime` and `DiscreteTime`, it has no dynamics. Choose it when the answer is a balance or an
    optimum, not a path.

    It appears in many fields. Flux balance analysis finds the steady fluxes of a metabolic network, an economy finds
    the least cost dispatch of power plants, and a circuit finds its operating point.

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
