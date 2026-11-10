"""Define the law that adds to the highest time derivative of a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from sage.structure.element import Matrix

from ...specs import RequirementSpec
from ..spaces import ContinuousTime
from ..types import Target
from ._base import BaseLaw


class RateLaw(BaseLaw):
    """A law that adds a rate to the highest time derivative of a state.

    On a state of order 1 the rate is its first derivative, and on a state of order `k` it adds to the `k`th derivative.
    Rates on the same state add up, so separate mechanisms superpose, and a system with only rate laws is a system of
    ordinary differential equations.

    Unlike `UpdateLaw`, which gives the next value in discrete time, it gives a derivative in continuous time. Choose it
    when a state flows, and add a `NoiseLaw` for a random part.

    It appears in many fields. Infected people recover at a rate, a balance grows by interest, a reactant is consumed by
    mass action, and a pendulum accelerates under gravity.

    Args:
        target:
            The symbol of a scalar state, the column of symbols of a vector state, or the matrix of a tensor state.

        value:
            A symbolic expression, a matrix of the shape of a tensor state, a `Call`, or a callable `f(t, values)` where
            `values` maps dotted state and parameter names to their current values.

        order:
            The order of the derivative the rate gives, checked against the state's order, or `None` to use it.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import RateLaw
        >>> n = var('n')
        >>> RateLaw(n, 2 * n).prepare().run({'n': 3.0}).value_
        6.0
    """

    law_tag: ClassVar[str] = 'rate'

    def __init__(
        self,
        target: Target | Matrix,
        value: object,
        order: int | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.value = value
        self.order = order

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the rate changes."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a value that is a NumPy array.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        self._check_array('value')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the rate at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.value, inputs)
