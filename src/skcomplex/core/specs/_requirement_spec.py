"""Define the requirement spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class RequirementSpec(BaseSpec):
    """A spec of what a component requires of the system it is part of.

    It holds the classes of time and of space it runs in. An empty field accepts any class.

    Args:
        time:
            The time classes the component runs in, such as `(ContinuousTime,)`.

        space:
            The space classes the component runs in, such as `(LatticeSpace,)`.

    Examples:
        >>> from skcomplex.core.components.spaces import ContinuousTime
        >>> from skcomplex.core.specs import RequirementSpec
        >>> RequirementSpec(time=(ContinuousTime,)).time
        (<class 'skcomplex.core.components.spaces._continuous_time.ContinuousTime'>,)
    """

    time: tuple[type, ...] = ()
    space: tuple[type, ...] = ()
