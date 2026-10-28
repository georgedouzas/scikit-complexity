"""Event compiled into a guard function and a reset function of the state vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from ...core.types import Vector
from ._base import BaseKernel
from ._state import State


class EventKernel(BaseKernel):
    """Event compiled into a guard function and a reset function of the state vector.

    Args:
        label:
            The name of the event.

        guard:
            The function `g(t, x, p)` whose crossing of zero triggers the event.

        reset:
            The function `r(t, x, p)` that returns the state vector after the event, or `None` to keep it.

        direction:
            `1`, `-1`, or `0` for crossings upwards, downwards, or both.

        terminal:
            Whether the event ends the run.

        hidden:
            Whether the event only marks a switch of a law at a condition on states, so integrators restart at the
            crossing, without a reset and without a record in the trajectory.
    """

    def __init__(  # noqa: PLR0917
        self,
        label: str,
        guard: Callable[[float, Vector, Vector], float],
        reset: Callable[[float, Vector, Vector], Vector] | None,
        direction: int,
        terminal: bool,
        hidden: bool = False,
    ) -> None:
        self.label = label
        self.guard = guard
        self.reset = reset
        self.direction = direction
        self.terminal = terminal
        self.hidden = hidden



    def _run(self, inputs: State) -> Vector:
        """Return the state vector after the event at the state, a copy of it without a reset."""
        if self.reset is None:
            return inputs.x.copy()
        return self.reset(inputs.t, inputs.x, inputs.p)
