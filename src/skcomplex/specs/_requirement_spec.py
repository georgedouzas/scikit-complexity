"""Define the requirement spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class RequirementSpec(BaseSpec):
    """What a part requires of the system it is part of: the classes of time and of space it runs in.

    An empty field accepts any class.

    Args:
        time:
            The time classes the part runs in, such as `(ContinuousTime,)`.

        space:
            The space classes the part runs in, such as `(LatticeSpace,)`.

    Examples:
        >>> from skcomplex.core.spaces import ContinuousTime
        >>> from skcomplex.specs import RequirementSpec
        >>> RequirementSpec(time=(ContinuousTime,)).time
        (<class 'skcomplex.core.spaces._continuous_time.ContinuousTime'>,)
    """

    time: tuple[type, ...] = ()
    space: tuple[type, ...] = ()
