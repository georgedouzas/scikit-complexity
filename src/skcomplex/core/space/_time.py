"""Define the time of a system: continuous, discrete with a schedule, or static."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..base import BaseEstimator, BaseTime, TimeKind, ValidationError

SCHEDULES = ('synchronous', 'random-sequential')


class Schedule(BaseEstimator):
    """Base class of a schedule of discrete time that a formalism defines, such as the phases of agents."""


class ContinuousTime(BaseTime):
    """Continuous time, in which rates, noise, implicit equations, jumps, and events change the states.

    Examples:
        >>> from skcomplex.core.space import ContinuousTime
        >>> str(ContinuousTime().prepare().time_kind_)
        'continuous'
    """

    kind: ClassVar[TimeKind] = TimeKind.CONTINUOUS


class DiscreteTime(BaseTime):
    """Discrete time, in which updates, rules, local rules, rewrites, and samples change the states every step.

    Args:
        schedule:
            `'synchronous'`, every update reads the states of the start of the step, `'random-sequential'`, members
            update one after another in a new random order, or a `Schedule` that a formalism defines, such as the
            phases of agents.

    Examples:
        >>> from skcomplex.core.space import DiscreteTime
        >>> time = DiscreteTime(schedule='random-sequential').prepare()
        >>> time.schedule_, time.run(3).tolist()
        ('random-sequential', [0.0, 1.0, 2.0, 3.0])
    """

    kind: ClassVar[TimeKind] = TimeKind.DISCRETE

    def __init__(self, schedule: str | Schedule = 'synchronous') -> None:
        self.schedule = schedule

    def _check(self) -> None:
        """Raise an error for a schedule that is neither known nor a `Schedule`."""
        if isinstance(self.schedule, Schedule):
            self.schedule.check()
            return
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}, or a Schedule.'
            raise ValidationError(error_msg)

    def _prepare(self) -> None:
        """Set the kind of the time and its schedule."""
        self.time_kind_ = self.kind
        self.schedule_ = self.schedule

    def _run(self, until: float) -> npt.NDArray[np.float64]:
        """Return every step from 0 to a time."""
        return np.arange(int(np.floor(until)) + 1, dtype=float)


class StaticTime(BaseTime):
    """No time: the system holds only constraints and objectives, and a run solves it once.

    Examples:
        >>> from skcomplex.core.space import StaticTime
        >>> StaticTime().prepare().run(10.0).tolist()
        [0.0]
    """

    kind: ClassVar[TimeKind] = TimeKind.STATIC

    def _run(self, until: float) -> npt.NDArray[np.float64]:
        """Return the one frame of the solution."""
        return np.zeros(1)
