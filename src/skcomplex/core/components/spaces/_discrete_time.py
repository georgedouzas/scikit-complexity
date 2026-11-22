"""Define discrete time, in which states change once per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ...errors import ValidationError
from ._base import _BaseTime
from ._schedule_spec import ScheduleSpec

SCHEDULES = ('synchronous', 'random-sequential')


class _DiscreteTime(_BaseTime):
    """A time that advances in steps, in which updates and rules change the states.

    The laws it accepts are updates, agent rules, local rules, rewrites, and Monte Carlo samples. A `schedule` may order
    the laws of one step in phases.

    Unlike `_ContinuousTime`, which flows, it advances by whole steps. Choose it when the model is a map, a cellular
    automaton, or an agent-based model.

    It appears in many fields. A population grows by generations, a cellular automaton steps its cells, agents trade in
    rounds, and an account compounds interest monthly.

    Args:
        schedule:
            `'synchronous'`, every update reads the states of the start of the step, `'random-sequential'`, members
            update one after another in a new random order, or a `ScheduleSpec` that a formalism defines, such as the
            phases of agents.

        label:
            The label of the time in its system.
    """

    time_tag: ClassVar[str] = 'discrete'

    def __init__(self, schedule: str | ScheduleSpec = 'synchronous', *, label: str = 'time') -> None:
        super().__init__(label)
        self.schedule = schedule

    def _prepare(self, context: None = None) -> None:
        """Set the schedule.

        It first raises an error for a schedule that is neither known nor a `ScheduleSpec`.

        Args:
            context:
                Nothing, since a time is prepared alone.
        """
        super()._prepare(context)
        if isinstance(self.schedule, ScheduleSpec):
            self.schedule._check()
        elif self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}, or a ScheduleSpec.'
            raise ValidationError(error_msg)
        self.schedule_ = self.schedule

