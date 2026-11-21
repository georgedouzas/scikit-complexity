"""Define the law that renews states after random waiting times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass

from sage.all__sagemath_symbolics import Expression

from ...components.laws import BaseLaw, RequirementSpec
from ...components.spaces._continuous_time import _ContinuousTime


@dataclass(frozen=True)
class RenewalLaw(BaseLaw):
    """A law that renews states after waiting times drawn anew at every renewal.

    The compiler runs it for a jump that waits on a clock, such as a walker that rests for a random time and then jumps.
    Every member of a group renews on its own clock. A renewal applies its reset to the states all at once, then draws
    the next waiting time from the states after the reset. Rate and noise laws move the states between renewals. It
    models the jumps of walkers, the failures and repairs of machines, and the firing of neurons.

    Read more in the user guide, under Laws.

    Args:
        waiting:
            The waiting time until the next renewal, usually a draw from a distribution with nonnegative support.

        reset:
            A mapping from state symbol to its new value, which may hold draws, or `None` to keep the states.

        label:
            The name of the renewal in the trajectory, or the empty string for the path of the element.

    See Also:
        JumpLaw: Changes states at random times.
        BoundRenewalSpec: Binds a renewal law to the group it renews.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import draw
        >>> from skcomplex.core.engine.compile._renewal_law import RenewalLaw
        >>> n = var('n')
        >>> RenewalLaw(draw('wait', 'expon'), {n: n + 1}).reset
        {n: n + 1}
    """

    waiting: object
    reset: Mapping[Expression, object] | None = None
    _: KW_ONLY
    label: str = ''

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return 'JumpLaw'

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
