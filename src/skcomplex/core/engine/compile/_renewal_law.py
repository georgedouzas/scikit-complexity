"""Define the renewal law the compiler runs for a jump on a clock of waiting times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from sage.all__sagemath_symbolics import Expression

from ...components.laws import BaseLaw, RequirementSpec
from ...components.spaces._continuous_time import _ContinuousTime


class RenewalLaw(BaseLaw):
    """A law that renews states after waiting times drawn anew at every renewal.

    The compiler runs it for a waiting jump. Every member of a group renews on its own clock. A renewal applies its
    reset to the states all at once, then draws the next waiting time from the states after the reset. Renewals combine
    with rate and noise laws, which evolve the states between renewals.

    Args:
        waiting:
            The waiting time until the next renewal, usually a draw from a distribution with nonnegative support.

        reset:
            A mapping from state symbol to its new value, which may hold draws, or `None` to keep the states.

        label:
            The name of the renewal in the trajectory, or the empty string for the path of the element.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import draw
        >>> from skcomplex.core.engine.compile._renewal_law import RenewalLaw
        >>> n = var('n')
        >>> RenewalLaw(draw('wait', 'expon'), {n: 0}).prepare().targets_
        (n,)
        >>> RenewalLaw(draw('wait', 'expon'), {n: n + 1}).reset
        {n: n + 1}
    """

    def __init__(self, waiting: object, reset: Mapping[Expression, object] | None = None, *, label: str = '') -> None:
        super().__init__(label)
        self.waiting = waiting
        self.reset = reset

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the renewal resets."""
        return tuple(self.reset or {})

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system.

        The law requires continuous time.
        """
        return RequirementSpec(time=(_ContinuousTime,))

    def _get_value(self, inputs: object) -> object:
        """Return the new values of the states at given values of the symbols of the reset.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        return self._evaluate(dict(self.reset or {}), inputs)
