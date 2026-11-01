"""Define the base of the time of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

from ..base import BasePart


class BaseTime(BasePart[None, SimpleNamespace]):
    """Base of the time of a system, the part that decides the laws the system may hold.

    A time is a part of its system: it stores its arguments in its own `__init__`, `check` validates them, `prepare`
    sets its schedule, and `_get_frames` returns the times of the frames of a run up to a time. It
    contributes no laws.

    Args:
        label:
            The label of the time in its system.

    Attributes:
        schedule_:
            The schedule of a discrete time, or `None`.
    """

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a time contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return no laws, since the time decides which laws a system may hold but contributes none."""
        return []

    def _check(self) -> None:
        """Accept the time, which has nothing to check besides its label by default."""

    def _get_frames(self, until: float) -> npt.NDArray[np.float64]:
        """Return the times of the frames of a run up to a time, its start and its end by default.

        Args:
            until:
                The end of the run.

        Returns:
            times:
                The times of the frames.
        """
        return np.array([0.0, float(until)])
