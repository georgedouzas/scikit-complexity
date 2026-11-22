"""Define the law that states an implicit relation between derivatives of states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ..spaces._continuous_time import _ContinuousTime
from ..spaces._static_time import _StaticTime
from ._base import BaseLaw
from ._requirement_spec import RequirementSpec


@dataclass(frozen=True)
class ConstraintLaw(BaseLaw):
    """A law that states an equation the compiler solves for derivatives of values.

    It gives an equation in place of a rate, such as `ConstraintLaw(s.l * s.theta.diff(s.t, 2), -s.g * sin(s.theta))`
    inside the interaction of a pendulum, which the compiler solves for the acceleration of the angle `theta`. The
    equation holds the highest derivatives of its values linearly, as Newton's second law holds an acceleration, and
    the compiler solves all the equations of a system together and adds the rate laws on the same values to the
    solution. The values are functions of time, such as `function('theta')(t)`. The equations of one `target` add into
    one before the solve, so a component may add a term without a highest derivative, such as a coupling. In static
    time it holds at the minimum of an `ObjectiveLaw`, where an inequality such as `v <= upper` bounds a value. The
    swings of a double pendulum, the currents of a circuit, and the joints of a robot arm follow such equations.

    Read more in the user guide, under Laws.

    Args:
        lhs:
            The left side, or a Sage relation `lhs == rhs`.

        rhs:
            The right side, or `None` when `lhs` is a relation.

        target:
            The value whose equation this is, so that the equations of one target add, or `None` for an equation of
            its own.

        label:
            The name of the law, or the empty string for a law without one.

    See Also:
        RateLaw: Gives a derivative of a value directly.
        ObjectiveLaw: Gives a cost to make least, under the equations of the system.

    Examples:
        >>> from sage.all__sagemath_symbolics import function, sin, var
        >>> from skcomplex.core.components.laws import ConstraintLaw
        >>> t, g, l = var('t g l')
        >>> theta = function('theta')(t)
        >>> ConstraintLaw(l * theta.diff(t, 2), -g * sin(theta)).rhs
        -g*sin(theta(t))
    """

    law_tag: ClassVar[str] = 'equation'

    lhs: object
    rhs: object = None
    _: KW_ONLY
    target: object = None
    label: str = ''

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states.

        The compiler finds the states whose derivatives the equation solves for.
        """
        return ()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous or static time."""
        return RequirementSpec(time=(_ContinuousTime, _StaticTime))

    def _get_sides(self) -> tuple[Expression, Expression]:
        """Return the two sides of the equation as expressions.

        It splits a relation given as the left side.
        """
        if self.rhs is None and isinstance(self.lhs, Expression):
            return self.lhs.lhs(), self.lhs.rhs()
        return SR(self.lhs), SR(self.rhs)

    def _check(self) -> None:
        """Raise an error for a function as a side, or a missing right side.

        An equation without a right side must be a relation.
        """
        super()._check()
        if any(callable(item) and not isinstance(item, Expression) for item in (self.lhs, self.rhs)):
            error_msg = (
                'Equations are solved symbolically, so a side of an equation cannot be a function. Use a `RateLaw` of '
                'the highest derivative with a function as its value.'
            )
            raise ValidationError(error_msg)
        if self.rhs is None and not (isinstance(self.lhs, Expression) and self.lhs.is_relational()):
            error_msg = 'An equation takes a left and a right side, or one relation `lhs == rhs`.'
            raise ValidationError(error_msg)

    def _get_value(self, inputs: object) -> object:
        """Return the residual `lhs - rhs` at given symbol values.

        The symbols are those it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            residual:
                The residual `lhs - rhs`.
        """
        lhs, rhs = self._get_sides()
        return self._evaluate(lhs - rhs, inputs)
