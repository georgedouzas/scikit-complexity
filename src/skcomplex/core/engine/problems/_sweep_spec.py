"""Define the sweep spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ...specs._base_spec import BaseSpec
from ...types import Array, Indices


@dataclass(frozen=True)
class _SweepSpec(BaseSpec):
    """Update law whose members update one after the other.

    Each member reads the values set before it.

    Args:
        indices:
            The flat indices of the state components every member sets, one row per member.

        key:
            The function `key(t, x, q)` of the order of the members, ascending.
    """

    indices: Indices
    key: Callable[[float, Array, Array], Array]
