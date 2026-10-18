"""Define the symbolic functions that laws use for conditions and remainders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

from sage.all__sagemath_symbolics import SR, floor
from sage.symbolic.function import BuiltinFunction

from ..core import Expression


def _is_decided(*arguments: Any) -> bool:  # noqa: ANN401
    """Check that arguments hold no symbols, so a condition on them is true or false."""
    return all(not SR(argument).variables() for argument in arguments)


class _Where(BuiltinFunction):
    """Value chosen by a condition, `where(condition, then, otherwise)`."""

    def __init__(self) -> None:
        super().__init__('where', nargs=3)

    def _eval_(self, condition: Expression, then: Expression, otherwise: Expression) -> Expression | None:
        """Return the chosen value when the condition is decided, or `None` to stay unevaluated."""
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


class _Derivative(BuiltinFunction):
    """Time derivative of a state of a given order, `derivative(x, j)`."""

    def __init__(self) -> None:
        super().__init__('derivative', nargs=2)

    def _eval_(self, value: Expression, order: Expression) -> Expression | None:
        """Return the state itself for order 0, or `None` to stay unevaluated."""
        if SR(order).is_numeric() and SR(order).pyobject() == 0:
            return value
        return None


class _Lag(BuiltinFunction):
    """Value of a state of a discrete map a given number of steps back, `lag(x, j)`."""

    def __init__(self) -> None:
        super().__init__('lag', nargs=2)

    def _eval_(self, value: Expression, steps: Expression) -> Expression | None:
        """Return the state itself for zero steps, or `None` to stay unevaluated."""
        if SR(steps).is_numeric() and SR(steps).pyobject() == 0:
            return value
        return None


where = _Where()
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
    >>> from skcomplex.core import symbol
    >>> from skcomplex.model import where
    >>> x = symbol('x')
    >>> where(x < 0, -x, x)
    where(x < 0, -x, x)
    >>> where(x < 0, -x, x).subs(x=-2)
    2
"""

either = _Either()
either.__doc__ = """Return a condition that holds where any of the conditions holds.

Args:
    *conditions:
        Relations or combinations of relations.

Returns:
    condition:
        The disjunction.

Examples:
    >>> from skcomplex.core import symbol
    >>> from skcomplex.model import either, where
    >>> x = symbol('x')
    >>> where(either(x < 0, x > 1), 1, 0).subs(x=2)
    1
"""

both = _Both()
both.__doc__ = """Return a condition that holds where all the conditions hold.

Args:
    *conditions:
        Relations or combinations of relations.

Returns:
    condition:
        The conjunction.

Examples:
    >>> from skcomplex.core import symbol
    >>> from skcomplex.model import both, where
    >>> x = symbol('x')
    >>> where(both(x > 0, x < 1), 1, 0).subs(x=0.5)
    1
"""

mod = _Mod()
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
    >>> from skcomplex.core import symbol
    >>> from skcomplex.model import mod
    >>> x = symbol('x')
    >>> mod(x, 1).subs(x=-0.25)
    0.75
"""

derivative = _Derivative()
derivative.__doc__ = """Return the time derivative of a state of a given order.

The compiler replaces derivatives below a state's order with its companion states `<name>_d<j>`, and an equation law
solves for the derivative of the state's own order.

Args:
    value:
        The symbol of a state.

    order:
        The order of the derivative, 0 for the state itself.

Returns:
    derivative:
        The symbolic derivative.

Examples:
    >>> from skcomplex.core import symbol
    >>> from skcomplex.model import derivative
    >>> x = symbol('x')
    >>> derivative(x, 2) + derivative(x, 0)
    x + derivative(x, 2)
"""

lag = _Lag()
lag.__doc__ = """Return the value of a state of a discrete map a given number of steps back.

The compiler replaces lags with the state's companion states `<name>_lag<j>`, which keep its past values.

Args:
    value:
        The symbol of a state.

    steps:
        The number of steps back, 0 for the current value.

Returns:
    lagged:
        The symbolic lagged value.

Examples:
    >>> from skcomplex.core import symbol
    >>> from skcomplex.model import lag
    >>> y = symbol('y')
    >>> lag(y, 1) - lag(y, 0)
    -y + lag(y, 1)
"""

__test__ = {
    'where': where.__doc__,
    'either': either.__doc__,
    'both': both.__doc__,
    'mod': mod.__doc__,
    'derivative': derivative.__doc__,
    'lag': lag.__doc__,
}
