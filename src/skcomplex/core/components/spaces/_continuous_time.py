"""Define continuous time, in which states flow by rates, noise, and events."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ._base import BaseTime


class ContinuousTime(BaseTime):
    """A time that flows continuously, in which rates and events change the states.

    The laws it accepts are rates, noise, implicit equations, random jumps, and crossings of a guard. A solver
    integrates them between the frames of a run.

    Unlike `DiscreteTime`, which steps, it lets states change at every instant. Choose it when the model is a
    differential equation or a stochastic process in continuous time.

    It appears in many fields. An epidemic flows by the SIR equations, a chemical reactor by its rate equations, a
    pendulum by Newton's law, and a stock price by a stochastic differential equation.

    Args:
        label:
            The label of the time in its system.

    Examples:
        >>> from skcomplex.core.components.spaces import ContinuousTime
        >>> ContinuousTime().prepare().schedule_ is None
        True
    """

    time_tag: ClassVar[str] = 'continuous'

    def __init__(self, *, label: str = 'time') -> None:
        super().__init__(label)

    def _prepare(self, context: None = None) -> None:
        """Set no schedule."""
        super()._prepare(context)
        self.schedule_ = None
