"""Define the spec of the order in which updates run within one step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...base import BaseSpec


class ScheduleSpec(BaseSpec):
    """An abstract spec of the order in which updates run within one step.

    A subclass orders the updates of a step of discrete time, such as the phases in which agents first move and then
    trade. It is a frozen dataclass of its own fields that overrides `_check`, and by default it accepts any schedule.
    Traders quote and then match orders, animals forage and then breed, and voters listen and then decide.

    Read more in the user guide, under Spaces.

    See Also:
        BaseSpace: Holds the members whose updates the schedule orders.

    Examples:
        >>> from skcomplex.core.base import BaseSpec
        >>> from skcomplex.core.components.spaces import ScheduleSpec
        >>> isinstance(ScheduleSpec(), BaseSpec)
        True
    """

    def _check(self) -> None:
        """Accept the schedule.

        By default it has nothing to check.
        """
