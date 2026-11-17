"""Define the relaxation spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ...specs._base_spec import BaseSpec
from ._base import BaseLaw


@dataclass(frozen=True)
class _RelaxationSpec(BaseSpec):
    """A spec of the relaxation that a call of `relax` names.

    Args:
        rules:
            The laws applied in turn until the state is stable.

        stable:
            The function of the arrays of the targets of the rules that returns whether the state is stable.

        drive:
            The laws applied once per step before the relaxation.

        activity:
            The function of the arrays of the targets that marks the active members of an iteration, or `None` to count
            the members that change.

        max_iterations:
            The largest number of iterations in one step.
    """

    rules: tuple[BaseLaw, ...]
    stable: Callable[..., bool]
    drive: tuple[BaseLaw, ...] = ()
    activity: Callable[..., object] | None = None
    max_iterations: int = 100_000
