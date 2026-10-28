"""Define the schedule spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from ._base_spec import BaseSpec


class ScheduleSpec(BaseSpec):
    """Base of the specs of schedules of discrete time that a formalism defines, such as the phases of agents.

    A subclass is a frozen dataclass of its own fields that overrides `check`.
    """

    def _check(self) -> Self:
        """Return the schedule, which has nothing to check by default.

        Returns:
            spec:
                The spec itself.
        """
        return self
