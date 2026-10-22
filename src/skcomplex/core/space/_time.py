"""Define the time of a system: continuous, discrete with a schedule, or static."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

from sklearn.base import BaseEstimator

from ..base import TimeKind, ValidationError
from ._phases import Phase

SCHEDULES = ('synchronous', 'random-sequential')


class Phases(BaseEstimator):
    """Schedule of a discrete time that runs named phases in order in every step.

    Args:
        phases:
            The phases, each with its rules and activation policy.

    Examples:
        >>> from skcomplex.core.space import Phase, Phases
        >>> [phase.name for phase in Phases([Phase('produce'), Phase('trade')]).phases]
        ['produce', 'trade']
    """

    def __init__(self, phases: Sequence[Phase]) -> None:
        self.phases = tuple(phases)
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            error_msg = f'The phases {names} repeat a name. Give every phase of a step its own name.'
            raise ValidationError(error_msg)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the names of the phases."""
        return f'Phases({[phase.name for phase in self.phases]!r})'


class Time(BaseEstimator):
    """Base class of the time of a system, which decides the laws the system may hold.

    Subclasses set the class variable `kind`.
    """

    kind: ClassVar[TimeKind]

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the name of the time."""
        return f'{type(self).__name__}()'


class ContinuousTime(Time):
    """Continuous time, in which rates, noise, implicit equations, jumps, and events change the states.

    Examples:
        >>> from skcomplex.core.space import ContinuousTime
        >>> str(ContinuousTime().kind)
        'continuous'
    """

    kind: ClassVar[TimeKind] = TimeKind.CONTINUOUS


class DiscreteTime(Time):
    """Discrete time, in which updates, rules, local rules, rewrites, and samples change the states every step.

    Args:
        schedule:
            `'synchronous'`, every update reads the states of the start of the step, `'random-sequential'`, members
            update one after another in a new random order, or `Phases`, named phases in order, each with its own
            activation policy.

    Examples:
        >>> from skcomplex.core.space import DiscreteTime, Phase, Phases
        >>> DiscreteTime(schedule=Phases([Phase('trade')])).schedule
        Phases(['trade'])
    """

    kind: ClassVar[TimeKind] = TimeKind.DISCRETE

    def __init__(self, schedule: str | Phases = 'synchronous') -> None:
        self.schedule = schedule
        if not isinstance(schedule, Phases) and schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {schedule!r}. Use one of {", ".join(SCHEDULES)}, or Phases.'
            raise ValidationError(error_msg)

    @property
    def phases(self) -> tuple[Phase, ...]:
        """Return the phases of the schedule, or none for a synchronous or random-sequential schedule."""
        return self.schedule.phases if isinstance(self.schedule, Phases) else ()

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the schedule."""
        return f'DiscreteTime(schedule={self.schedule!r})'


class StaticTime(Time):
    """No time: the system holds only constraints and objectives, and a run solves it once.

    Examples:
        >>> from skcomplex.core.space import StaticTime
        >>> str(StaticTime().kind)
        'static'
    """

    kind: ClassVar[TimeKind] = TimeKind.STATIC
