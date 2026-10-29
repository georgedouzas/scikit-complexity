"""Define crossing laws, events at the crossing of zero by a guard."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime
from ._base import BaseLaw


class CrossingLaw(BaseLaw):
    """Event at the crossing of zero by a guard, when states take new values or the run ends.

    Solvers locate the crossing within their tolerance. Crossings leave the form of a system as is, so they combine with
    rate laws, and the trajectory records every crossing with its time and the states before and after it.

    Args:
        guard:
            A symbolic expression of states and time whose crossing of zero triggers the event.

        reset:
            A mapping from state symbol to its new value, applied all at once, or `None` to keep the states.

        direction:
            `1` for crossings from negative to positive, `-1` for the opposite, and `0` for both.

        terminal:
            Whether the crossing ends the run.

        label:
            The name of the crossing in the trajectory, or the empty string for the path of the element.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import CrossingLaw
        >>> height, velocity = var('height'), var('velocity')
        >>> CrossingLaw(height, {velocity: -0.8 * velocity}, direction=-1).check().run({'velocity': -2.0}).value_
        {velocity: 1.6}
    """

    def __init__(
        self,
        guard: object,
        reset: Mapping[Expression, object] | None = None,
        direction: int = 0,
        terminal: bool = False,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.guard = guard
        self.reset = reset
        self.direction = direction
        self.terminal = terminal

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the crossing resets."""
        return tuple(self.reset or {})

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _check(self) -> None:
        """Raise an error for a direction that is not -1, 0, or 1."""
        if self.direction not in {-1, 0, 1}:
            error_msg = f'The direction of a crossing is -1, 0, or 1, but got {self.direction!r}.'
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_` to the new values of the states at given values of the symbols the reset reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(dict(self.reset or {}), inputs)
