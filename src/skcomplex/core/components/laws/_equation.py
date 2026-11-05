"""Define equation laws, implicit relations in which the highest time derivatives of states appear."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime, StaticTime
from ._base import BaseLaw


class EquationLaw(BaseLaw):
    """Implicit relation in which the highest time derivatives of states appear, such as Newton's second law.

    The states are Sage functions of time, such as `function('theta')(t)`, and their derivatives are Sage derivatives,
    `theta.diff(t, 2)`. The compiler solves all equations of a system jointly for the highest derivatives, which must
    appear linearly, and adds the rate laws on the same states to the solution. An equation may name the state whose
    equation it is, its `target`, and the equations of one target, from any element or interaction, add into one
    before the solve, so a term of it may hold no highest derivative, such as the part a coupling Lagrangian adds to
    the Euler-Lagrange equation of a coordinate.

    Args:
        lhs:
            The left side, or a Sage relation `lhs == rhs`.

        rhs:
            The right side, or `None` when `lhs` is a relation.

        target:
            The state whose equation this is, so that the equations of one target add, or `None` for an equation of
            its own.

        label:
            The name of the law, or the empty string for a law without one.

    Attributes:
        lhs_:
            The left side.

        rhs_:
            The right side.

    Examples:
        >>> from sage.all__sagemath_symbolics import function, sin, var
        >>> from skcomplex.core.components.laws import EquationLaw
        >>> t, g, l = var('t g l')
        >>> theta = function('theta')(t)
        >>> EquationLaw(l * theta.diff(t, 2) == -g * sin(theta)).prepare().rhs_
        -g*sin(theta(t))
    """

    def __init__(self, lhs: object, rhs: object = None, *, target: object = None, label: str = '') -> None:
        super().__init__(label)
        self.lhs = lhs
        self.rhs = rhs
        self.target = target

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since the compiler finds the states whose derivatives the equation solves for."""
        return ()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous or static time."""
        return RequirementSpec(time=(ContinuousTime, StaticTime))

    def _get_sides(self) -> tuple[Expression, Expression]:
        """Return the two sides of the equation as expressions, splitting a relation given as the left side."""
        if self.rhs is None and isinstance(self.lhs, Expression):
            return self.lhs.lhs(), self.lhs.rhs()
        return SR(self.lhs), SR(self.rhs)

    def _check(self) -> None:
        """Raise an error for a function as a side, or an equation without a right side that is not a relation."""
        if any(callable(item) and not isinstance(item, Expression) for item in (self.lhs, self.rhs)):
            error_msg = (
                'Equations are solved symbolically, so a side of an equation cannot be a function. Use a `RateLaw` of '
                'the highest derivative with a function as its value.'
            )
            raise ValidationError(error_msg)
        if self.rhs is None and not (isinstance(self.lhs, Expression) and self.lhs.is_relational()):
            error_msg = 'An equation takes a left and a right side, or one relation `lhs == rhs`.'
            raise ValidationError(error_msg)

    def _prepare(self, context: str | None = None) -> None:
        """Set `lhs_` and `rhs_`, the two sides, besides the attributes of every law.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        self.lhs_, self.rhs_ = self._get_sides()

    def _run(self, inputs: object) -> None:
        """Set `value_` to the residual `lhs - rhs` at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        lhs, rhs = self._get_sides()
        self.value_ = self._evaluate(lhs - rhs, inputs)
