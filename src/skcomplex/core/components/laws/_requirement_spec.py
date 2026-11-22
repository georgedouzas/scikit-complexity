"""Define the requirement spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec


@dataclass(frozen=True)
class RequirementSpec(BaseSpec):
    """A spec of the time and the space that a law needs of a system.

    It names the classes of time and of space a law runs in, such as the continuous time of `RateLaw.get_requirements()`
    or the discrete time of an `UpdateLaw`, and an empty field accepts any class. A law class returns it before any law
    exists, so a system can say which laws fit its time and its space. With `sites`, a law that reads sites also runs
    on any other space with sites, such as the graph of a group that a system package defines. A chemical reaction
    needs continuous time, a cellular automaton a lattice, and a voter model any space of sites.

    Read more in the user guide, under Laws.

    Args:
        time:
            The classes of time the law runs in, such as continuous time alone.

        space:
            The classes of space the law runs in, such as `(LatticeSpace,)`.

        sites:
            Whether the law also runs on any other space with sites.

    See Also:
        BaseLaw: Returns the spec of its class from `get_requirements`.

    Examples:
        >>> from skcomplex.core.components.laws import RequirementSpec
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> needs = RequirementSpec(space=(LatticeSpace,), sites=True)
        >>> needs.sites, needs.time
        (True, ())
    """

    time: tuple[type, ...] = ()
    space: tuple[type, ...] = ()
    sites: bool = False
