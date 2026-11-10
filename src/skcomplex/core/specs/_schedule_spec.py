"""Define the schedule spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ._base_spec import BaseSpec


class ScheduleSpec(BaseSpec):
    """An abstract spec of a schedule of discrete time.

    An example is the phases of agents. A subclass is a frozen dataclass of its own fields that overrides `_check`.
    """

    def _check(self) -> None:
        """Accept the schedule.

        By default it has nothing to check.
        """
