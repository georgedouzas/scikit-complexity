"""Define the requirement spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec


@dataclass(frozen=True)
class RequirementSpec(BaseSpec):
    """A spec of what a component requires of the system it is part of.

    It holds the classes of time and of space it runs in. An empty field accepts any class. A component that reads
    sites may also accept any other space with sites, such as the Cayley graph of a group that a system package
    defines.

    Args:
        time:
            The time classes the component runs in, such as continuous time alone.

        space:
            The space classes the component runs in, such as `(LatticeSpace,)`.

        sites:
            Whether the component also runs on any other space with sites, one whose `_has_sites` is true.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.components.laws import RequirementSpec
        >>> RequirementSpec(space=(LatticeSpace,)).space
        (<class 'skcomplex.core.components.spaces._lattice.LatticeSpace'>,)
        >>> RequirementSpec(space=(LatticeSpace,), sites=True).sites
        True
    """

    time: tuple[type, ...] = ()
    space: tuple[type, ...] = ()
    sites: bool = False
