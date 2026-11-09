"""Define the symbolic conditions and remainders that laws use.

Every backend lowers them.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression, floor
from sage.symbolic.function import BuiltinFunction

from ._draws import draws_in


def _is_number(argument: object) -> bool:
    """Check that an expression, or both sides of a relation, evaluate to numbers.

    So no call stays unevaluated.
    """
    value = SR(argument)
    sides = [value.lhs(), value.rhs()] if value.is_relational() else [value]
    try:
        for side in sides:
            complex(side)
    except (TypeError, ValueError):
        return False
    return True


def _is_decided(*arguments: object) -> bool:
    """Check that arguments hold no symbols, random draws, or unevaluated calls.

    So the condition is decided.
    """
    return all(
        not SR(argument).variables() and not draws_in(argument) and _is_number(argument) for argument in arguments
    )


class _WhereFunction(BuiltinFunction):
    """Value chosen by a condition, `where(condition, then, otherwise)`."""

    def __init__(self) -> None:
        super().__init__('where', nargs=3)

    def _eval_(self, condition: Expression, then: Expression, otherwise: Expression) -> Expression | None:
        """Return the chosen value when it is decided, or `None` otherwise.

        The value is decided when the condition is decided or both values are equal.
        """
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
        """Return the derivative of the chosen value.

        It is zero at the switch in the sense of distributions.
        """
        return self(condition, SR(then).diff(diff_param), SR(otherwise).diff(diff_param))


class _EitherFunction(BuiltinFunction):
    """Disjunction of conditions, true when any of them is true."""

    def __init__(self) -> None:
        super().__init__('either', nargs=0)

    def _eval_(self, *conditions: Expression) -> Expression | None:
        """Return 1 or 0 when the conditions are decided, or `None` to stay unevaluated."""
        if _is_decided(*conditions):
            return SR(int(any(bool(condition) for condition in conditions)))
        return None


class _BothFunction(BuiltinFunction):
    """Conjunction of conditions, true when all of them are true."""

    def __init__(self) -> None:
        super().__init__('both', nargs=0)

    def _eval_(self, *conditions: Expression) -> Expression | None:
        """Return 1 or 0 when the conditions are decided, or `None` to stay unevaluated."""
        if _is_decided(*conditions):
            return SR(int(all(bool(condition) for condition in conditions)))
        return None


class _ModFunction(BuiltinFunction):
    """Remainder of a division with the sign of the divisor, `mod(value, divisor)`."""

    def __init__(self) -> None:
        super().__init__('mod', nargs=2)

    def _eval_(self, value: Expression, divisor: Expression) -> Expression | None:
        """Return the remainder when both arguments are numbers.

        It returns `None` to stay unevaluated.
        """
        if _is_decided(value, divisor):
            return value - divisor * floor(value / divisor)
        return None

    def _tderivative_(self, value: Expression, divisor: Expression, diff_param: Expression) -> Expression:
        """Return the derivative away from the jumps.

        It is `d(value) - floor(value / divisor) d(divisor)`.
        """
        return SR(value).diff(diff_param) - floor(value / divisor) * SR(divisor).diff(diff_param)


where = _WhereFunction()


where.__doc__ = """Return `then` where a condition holds and `otherwise` elsewhere.

The value stays symbolic while the condition has symbols, and every backend lowers it. Combine conditions with
`either` and `both`.

Args:
    condition:
        A relation such as `x < 0`, or a combination of relations.

    then:
        The value where the condition holds.

    otherwise:
        The value elsewhere.

Returns:
    value:
        The symbolic choice.

Examples:
    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws.utils import where
    >>> x = var('x')
    >>> where(x < 0, -x, x)
    where(x < 0, -x, x)
    >>> where(x < 0, -x, x).subs(x=-2)
    2
"""


either = _EitherFunction()


either.__doc__ = """Return a condition that holds where any of the conditions holds.

Args:
    *conditions:
        Relations or combinations of relations.

Returns:
    condition:
        The disjunction.

Examples:
    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws.utils import either, where
    >>> x = var('x')
    >>> where(either(x < 0, x > 1), 1, 0).subs(x=2)
    1
"""


both = _BothFunction()


both.__doc__ = """Return a condition that holds where all the conditions hold.

Args:
    *conditions:
        Relations or combinations of relations.

Returns:
    condition:
        The conjunction.

Examples:
    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws.utils import both, where
    >>> x = var('x')
    >>> where(both(x > 0, x < 1), 1, 0).subs(x=0.5)
    1
"""


mod = _ModFunction()


mod.__doc__ = """Return the remainder of a division, with the sign of the divisor as Python's `%`.

Args:
    value:
        The dividend.

    divisor:
        The divisor.

Returns:
    remainder:
        The symbolic remainder.

Examples:
    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws.utils import mod
    >>> x = var('x')
    >>> mod(x, 1).subs(x=-0.25)
    0.75
"""
