"""Event compiled into a guard function and a reset function of the state vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from ...specs import RunStateSpec
from ...types import Array
from ._base import BaseKernel


class EventKernel(BaseKernel):
    """A kernel of an event, compiled into a guard and a reset of the state vector.

    Args:
        label:
            The name of the event.

        guard:
            The function `g(t, x, p)` whose crossing of zero triggers the event, or for an event of the members of a
            group, whose value for every member triggers the event of that member.

        reset:
            The function `r(t, x, p)` that returns the state vector after the event, or for an event of the members of
            a group, `r(t, x, p, member)` that returns it after the event of one member, or `None` to keep it.

        direction:
            `1`, `-1`, or `0` for crossings upwards, downwards, or both.

        terminal:
            Whether the event ends the run.

        hidden:
            Whether the event only marks a switch of a law at a condition on states, so integrators restart at the
            crossing, without a reset and without a record in the trajectory.

        members:
            The number of members of the group whose members each have the event, or 0 for one event.

    Attributes:
        values_:
            The state vector after the event at the state of the last run.
    """

    values_: Array

    def __init__(
        self,
        label: str,
        guard: Callable[[float, Array, Array], float | Array],
        reset: Callable[..., Array] | None,
        direction: int,
        terminal: bool,
        *,
        hidden: bool = False,
        members: int = 0,
    ) -> None:
        self.label = label
        self.guard = guard
        self.reset = reset
        self.direction = direction
        self.terminal = terminal
        self.hidden = hidden
        self.members = members

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the state vector after the event at the state, a copy of it without a reset."""
        state = self._get_state_of(inputs)
        self.values_ = state.x.copy() if self.reset is None else self.reset(state.t, state.x, state.p)
