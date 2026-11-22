"""Define the law that fires an event when a guard crosses zero."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ..spaces._continuous_time import _ContinuousTime
from ._base import BaseLaw
from ._requirement_spec import RequirementSpec


@dataclass(frozen=True)
class EventLaw(BaseLaw):
    """A law that sets values or ends the run when an expression crosses zero.

    It watches an expression of the values and acts when it crosses zero, such as `EventLaw(s.ball.x, {s.ball.v: -0.8 *
    s.ball.v}, direction=-1)` inside the interaction of a bouncing ball, which turns and slows the speed `v` when the
    height `x` falls through zero. At the crossing the values of `reset` take their new values, or the run ends when
    `terminal` is true. The solver finds the time of the crossing within its tolerance, which a step of an `UpdateLaw`
    cannot, and the trajectory records every crossing with its time and the values before and after it. The rate laws
    of the system stay as they are. A ball bounces off the floor, a neuron fires when its voltage reaches a threshold,
    and a thermostat switches a heater at a set temperature.

    Read more in the user guide, under Laws.

    Args:
        guard:
            A symbolic expression of values and time whose crossing of zero sets off the event, a function
            `f(t, values)` of time and the mapping from the dotted name of every value and parameter to its value, or
            `CallSpec(f, relative=True)` to read the names of the component. Over a group it returns one value per
            member. It runs in Python, and a function reads no derivatives or past values.

        reset:
            A mapping from the symbol of a value to its new value, all set at once, or `None` to keep the values. A
            new value may be a function like the guard.

        direction:
            `1` for crossings from negative to positive, `-1` for the opposite, and `0` for both.

        terminal:
            Whether the crossing ends the run.

        label:
            The name of the crossing in the trajectory, or the empty string for the path of the element.

    See Also:
        JumpLaw: Changes values at random times.
        UpdateLaw: Gives the next value after one step of discrete time.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import EventLaw
        >>> x, v = var('x v')
        >>> EventLaw(x, {v: -0.8 * v}, direction=-1).reset
        {v: -0.8*v}
    """

    law_tag: ClassVar[str] = 'update'

    guard: object
    reset: Mapping[Expression, object] | None = None
    direction: int = 0
    terminal: bool = False
    _: KW_ONLY
    label: str = ''

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the crossing resets."""
        return tuple(self.reset or {})

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(_ContinuousTime,))

    def _check(self) -> None:
        """Raise an error for a direction that is not -1, 0, or 1."""
        super()._check()
        if self.direction not in {-1, 0, 1}:
            error_msg = f'The direction of a crossing is -1, 0, or 1, but got {self.direction!r}.'
            raise ValidationError(error_msg)

    def _get_value(self, inputs: object) -> object:
        """Return the new values at given values of the symbols the reset reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            values:
                The new value of every state the reset names.
        """
        return self._evaluate(dict(self.reset or {}), inputs)
