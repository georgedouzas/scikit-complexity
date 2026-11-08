"""Define crossing laws, events at the crossing of zero by a guard."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime
from ._base import BaseLaw


class CrossingLaw(BaseLaw):
    """Event at the crossing of zero by a guard.

    At the event, states take new values or the run ends. Solvers locate the crossing within their tolerance. Crossings
    leave the form of a system as is, so they combine with rate laws, and the trajectory records every crossing with its
    time and the states before and after it.

    Args:
        guard:
            A symbolic expression of states and time whose crossing of zero triggers the event, or a function
            `f(t, values)` of time and the mapping from the dotted name of every state and parameter to its value, or
            `CallSpec(f, relative=True)` to read the names of the component. Over a group it returns one value per
            member. It runs in Python, and a function does not lower the order of derivatives or read delays.

        reset:
            A mapping from state symbol to its new value, applied all at once, or `None` to keep the states. A value
            may be a function like the guard's.

        direction:
            `1` for crossings from negative to positive, `-1` for the opposite, and `0` for both.

        terminal:
            Whether the crossing ends the run.

        label:
            The name of the crossing in the trajectory, or the empty string for the path of the element.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import CrossingLaw
        >>> y, z = var('y'), var('z')
        >>> law = CrossingLaw(y, {z: -0.8 * z}, direction=-1).prepare()
        >>> law.run({'z': -2.0}).value_
        {z: 1.6}
    """

    law_tag: ClassVar[str] = 'update'

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

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _check(self) -> None:
        """Raise an error for a direction that is not -1, 0, or 1."""
        if self.direction not in {-1, 0, 1}:
            error_msg = f'The direction of a crossing is -1, 0, or 1, but got {self.direction!r}.'
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_` to the new values of the states at given symbol values.

        The symbols are those the reset reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(dict(self.reset or {}), inputs)
