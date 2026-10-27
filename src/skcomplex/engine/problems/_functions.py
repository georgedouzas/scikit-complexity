"""Define the symbolic conditions and remainders that the compiler lowers laws into, and that backends print."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import SR, Expression, floor
from sage.symbolic.function import BuiltinFunction

from ...core.laws.utils import draws_in


def _is_number(argument: object) -> bool:
    """Check that an expression, or both sides of a relation, evaluate to numbers, so no call stays unevaluated."""
    value = SR(argument)
    sides = [value.lhs(), value.rhs()] if value.is_relational() else [value]
    try:
        for side in sides:
            complex(side)
    except (TypeError, ValueError):
        return False
    return True


def _is_decided(*arguments: object) -> bool:
    """Check that arguments hold no symbols, no random draws, and no unevaluated calls, so a condition is decided."""
    return all(
        not SR(argument).variables() and not draws_in(argument) and _is_number(argument) for argument in arguments
    )


class _Where(BuiltinFunction):
    """Value chosen by a condition, `where(condition, then, otherwise)`."""

    def __init__(self) -> None:
        super().__init__('where', nargs=3)

    def _eval_(self, condition: Expression, then: Expression, otherwise: Expression) -> Expression | None:
        """Return the chosen value when the condition is decided or both values are equal, or `None` otherwise."""
        if SR(then).is_trivially_equal(SR(otherwise)):
            return SR(then)
        if _is_decided(condition):
            return then if bool(condition) else otherwise
        return None

    def _tderivative_(
        self,
        condition: Expression,
        then: Expression,
        otherwise: Expression,
        diff_param: Expression,
    ) -> Expression:
        """Return the derivative of the chosen value, which is zero at the switch in the sense of distributions."""
        return self(condition, SR(then).diff(diff_param), SR(otherwise).diff(diff_param))


class _Either(BuiltinFunction):
    """Disjunction of conditions, true when any of them is true."""

    def __init__(self) -> None:
        super().__init__('either', nargs=0)

    def _eval_(self, *conditions: Expression) -> Expression | None:
        """Return 1 or 0 when the conditions are decided, or `None` to stay unevaluated."""
        if _is_decided(*conditions):
            return SR(int(any(bool(condition) for condition in conditions)))
        return None


class _Both(BuiltinFunction):
    """Conjunction of conditions, true when all of them are true."""

    def __init__(self) -> None:
        super().__init__('both', nargs=0)

    def _eval_(self, *conditions: Expression) -> Expression | None:
        """Return 1 or 0 when the conditions are decided, or `None` to stay unevaluated."""
        if _is_decided(*conditions):
            return SR(int(all(bool(condition) for condition in conditions)))
        return None


class _Mod(BuiltinFunction):
    """Remainder of a division with the sign of the divisor, `mod(value, divisor)`."""

    def __init__(self) -> None:
        super().__init__('mod', nargs=2)

    def _eval_(self, value: Expression, divisor: Expression) -> Expression | None:
        """Return the remainder when both arguments are numbers, or `None` to stay unevaluated."""
        if _is_decided(value, divisor):
            return value - divisor * floor(value / divisor)
        return None

    def _tderivative_(self, value: Expression, divisor: Expression, diff_param: Expression) -> Expression:
        """Return the derivative away from the jumps, `d(value) - floor(value / divisor) d(divisor)`."""
        return SR(value).diff(diff_param) - floor(value / divisor) * SR(divisor).diff(diff_param)


where = _Where()
"""Value `then` where a condition holds and `otherwise` elsewhere, `where(condition, then, otherwise)`."""

either = _Either()
"""Disjunction of conditions, true when any of them is true."""

both = _Both()
"""Conjunction of conditions, true when all of them are true."""

mod = _Mod()
"""Remainder of a division with the sign of the divisor, `mod(value, divisor)`."""
