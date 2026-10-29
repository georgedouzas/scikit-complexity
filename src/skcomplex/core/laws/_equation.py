"""Define equation laws, implicit relations in which the highest time derivatives of states appear."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime, StaticTime
from ._base import BaseLaw


class EquationLaw(BaseLaw):
    """Implicit relation in which the highest time derivatives of states appear, such as Newton's second law.

    The states are Sage functions of time, such as `function('theta')(t)`, and their derivatives are Sage derivatives,
    `theta.diff(t, 2)`. The compiler solves all equations of a system jointly for the highest derivatives, which must
    appear linearly, and adds the rate laws on the same states to the solution.

    Args:
        lhs:
            The left side, or a Sage relation `lhs == rhs`.

        rhs:
            The right side, or `None` when `lhs` is a relation.

        label:
            The name of the law, or the empty string for a law without one.

    Attributes:
        lhs_:
            The left side.

        rhs_:
            The right side.

    Examples:
        >>> from sage.all__sagemath_symbolics import function, sin, var
        >>> from skcomplex.core.laws import EquationLaw
        >>> t, g, l = var('t g l')
        >>> theta = function('theta')(t)
        >>> EquationLaw(l * theta.diff(t, 2) == -g * sin(theta)).check().prepare().rhs_
        -g*sin(theta(t))
    """

    def __init__(self, lhs: object, rhs: object = None, *, label: str = '') -> None:
        super().__init__(label)
        self.lhs = lhs
        self.rhs = rhs

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since the compiler finds the states whose derivatives the equation solves for."""
        return ()

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous or static time."""
        return RequirementSpec(time=(ContinuousTime, StaticTime))

    def _get_sides(self) -> tuple[Expression, Expression]:
        """Return the two sides of the equation as expressions, splitting a relation given as the left side."""
        if self.rhs is None and isinstance(self.lhs, Expression):
            return self.lhs.lhs(), self.lhs.rhs()
        return SR(self.lhs), SR(self.rhs)

    def _check(self) -> Self:
        """Raise an error for an equation without a right side whose left side is not a relation.

        Returns:
            law:
                The law itself.
        """
        if self.rhs is None and not (isinstance(self.lhs, Expression) and self.lhs.is_relational()):
            error_msg = 'An equation takes a left and a right side, or one relation `lhs == rhs`.'
            raise ValidationError(error_msg)
        return self

    def _prepare(self, context: str | None = None) -> Self:
        """Set `lhs_` and `rhs_`, the two sides, besides the attributes of every law.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.

        Returns:
            law:
                The law itself.
        """
        super()._prepare(context)
        self.lhs_, self.rhs_ = self._get_sides()
        return self

    def _run(self, inputs: object) -> object:
        """Return the residual `lhs - rhs` at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            residual:
                The residual, a number when nothing symbolic is left.
        """
        lhs, rhs = self._get_sides()
        return self._evaluate(lhs - rhs, inputs)
