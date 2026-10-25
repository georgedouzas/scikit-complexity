"""Define the time of a system: continuous, discrete with a schedule, or static."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from sklearn.base import BaseEstimator

from ..base import BaseTime, TimeKind, ValidationError

SCHEDULES = ('synchronous', 'random-sequential')


class Schedule(BaseEstimator):
    """Base class of a schedule of discrete time that a formalism defines, such as the phases of agents."""

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the name of the schedule."""
        return f'{type(self).__name__}()'


class ContinuousTime(BaseTime):
    """Continuous time, in which rates, noise, implicit equations, jumps, and events change the states.

    Examples:
        >>> from skcomplex.core.space import ContinuousTime
        >>> str(ContinuousTime().kind)
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
        >>> DiscreteTime(schedule='random-sequential').schedule
        'random-sequential'
    """

    kind: ClassVar[TimeKind] = TimeKind.DISCRETE

    def __init__(self, schedule: str | Schedule = 'synchronous') -> None:
        self.schedule = schedule
        if not isinstance(schedule, Schedule) and schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {schedule!r}. Use one of {", ".join(SCHEDULES)}, or a Schedule.'
            raise ValidationError(error_msg)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the schedule."""
        return f'DiscreteTime(schedule={self.schedule!r})'


class StaticTime(BaseTime):
    """No time: the system holds only constraints and objectives, and a run solves it once.

    Examples:
        >>> from skcomplex.core.space import StaticTime
        >>> str(StaticTime().kind)
        'static'
    """

    kind: ClassVar[TimeKind] = TimeKind.STATIC
