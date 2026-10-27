"""Define the renewal law the compiler runs for a jump on a clock of waiting times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Self

from sage.all__sagemath_symbolics import Expression

from ...core.laws import BaseLaw
from ...core.spaces import ContinuousTime
from ...core.specs import RequirementSpec


class RenewalLaw(BaseLaw):
    """Renewal of states after waiting times drawn anew at every renewal, which the compiler runs for a waiting jump.

    Every member of a group renews on its own clock. A renewal applies its reset to the states all at once, then draws
    the next waiting time from the states after the reset. Renewals combine with rate and noise laws, which evolve the
    states between renewals.

    Args:
        waiting:
            The waiting time until the next renewal, usually a draw from a distribution with nonnegative support.

        reset:
            A mapping from state symbol to its new value, which may hold draws, or `None` to keep the states.

        label:
            The name of the renewal in the trajectory, or the empty string for the path of the element.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import draw
        >>> from skcomplex.engine.compile import RenewalLaw
        >>> n = var('n')
        >>> RenewalLaw(draw('wait', 'expon'), {n: 0}).check().prepare().targets_
        (n,)
    """

    def __init__(self, waiting: object, reset: Mapping[Expression, object] | None = None, *, label: str = '') -> None:
        super().__init__(label)
        self.waiting = waiting
        self.reset = reset

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the renewal resets."""
        return tuple(self.reset or {})

    def _check(self) -> Self:
        """Return the law, whose waiting time and reset the jump it is run for checked.

        Returns:
            law:
                The law itself.
        """
        return self

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _run(self, inputs: object) -> object:
        """Return the new values of the states at given values of the symbols the reset reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            reset:
                The new values by state.
        """
        return self._evaluate(dict(self.reset or {}), inputs)
