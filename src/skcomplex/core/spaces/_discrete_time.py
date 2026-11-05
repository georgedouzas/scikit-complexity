"""Define discrete time, in which updates and rules change the states every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ...errors import ValidationError
from ...specs import ScheduleSpec
from ._base import BaseTime

SCHEDULES = ('synchronous', 'random-sequential')


class DiscreteTime(BaseTime):
    """Discrete time, in which updates, rules, local rules, rewrites, and samples change the states every step.

    Args:
        schedule:
            `'synchronous'`, every update reads the states of the start of the step, `'random-sequential'`, members
            update one after another in a new random order, or a `ScheduleSpec` that a formalism defines, such as the
            phases of agents.

        label:
            The label of the time in its system.

    Examples:
        >>> from skcomplex.core.spaces import DiscreteTime
        >>> DiscreteTime(schedule='random-sequential').prepare().schedule_
        'random-sequential'
    """

    def __init__(self, schedule: str | ScheduleSpec = 'synchronous', *, label: str = 'time') -> None:
        super().__init__(label)
        self.schedule = schedule

    def _check(self) -> None:
        """Raise an error for a schedule that is neither known nor a `ScheduleSpec`."""
        if isinstance(self.schedule, ScheduleSpec):
            self.schedule._validated()
            return
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}, or a ScheduleSpec.'
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Set the schedule."""
        self.schedule_ = self.schedule

    def _get_frames(self, until: float) -> npt.NDArray[np.float64]:
        """Return every step from 0 to a time."""
        return np.arange(int(np.floor(until)) + 1, dtype=float)
