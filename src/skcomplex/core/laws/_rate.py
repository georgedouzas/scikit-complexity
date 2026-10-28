"""Define rate laws, contributions to the highest time derivative of a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from ...specs import RequirementSpec
from ...types import Target
from ..spaces import ContinuousTime
from ._base import BaseLaw


class RateLaw(BaseLaw):
    """Contribution to the highest time derivative of a state, the first for states of order 1.

    Rates on the same state add up, so a system with only rate laws is a system of ordinary differential equations. On
    a state of order `k`, a rate contributes to the `k`-th derivative, so independent mechanisms superpose.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression, a `Call`, or a callable `f(t, values)` where `values` maps dotted state and parameter
            names to their current values.

        order:
            The order of the derivative the rate gives, checked against the state's order, or `None` to use it.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import RateLaw
        >>> n = var('n')
        >>> RateLaw(n, 2 * n).check().run({'n': 3.0})
        6.0
    """

    def __init__(self, target: Target, value: object, order: int | None = None, *, label: str = '') -> None:
        super().__init__(label)
        self.target = target
        self.value = value
        self.order = order

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the rate changes."""
        return (self.target,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _check(self) -> Self:
        """Raise an error for a value that is a NumPy array.

        Returns:
            law:
                The law itself.
        """
        self._check_array('value')
        return self

    def _run(self, inputs: object) -> object:
        """Return the rate at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            rate:
                The rate, a number when nothing symbolic is left.
        """
        return self._evaluate(self.value, inputs)
