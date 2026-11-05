"""Define the jump path spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class JumpPathSpec(BaseSpec):
    """Path of a discrete state with the time of every jump.

    Args:
        times:
            The first time, then the time of every frame at which the state changed.

        states:
            The state at the first time and after every jump, one row per jump for vector states.

        label:
            The name of the state.
    """

    times: Array
    states: Array
    label: str
