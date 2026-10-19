"""Define the symbolic functions that laws use for conditions and remainders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from typing import Any

from sage.all__sagemath_symbolics import SR, floor
from sage.symbolic.function import BuiltinFunction

from ..base import Expression


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


class _Delayed(BuiltinFunction):
    """Value of a state a delay before the current time, `delayed(x, tau)` or `delayed(x, tau, minimum)`."""

    def __init__(self) -> None:
        super().__init__('delayed', nargs=0)

    def _eval_(self, *arguments: Expression) -> Expression | None:
        """Return the state itself for a delay of 0, or `None` to stay unevaluated."""
        if len(arguments) >= 2 and SR(arguments[1]).is_numeric() and SR(arguments[1]).pyobject() == 0:  # noqa: PLR2004
            return arguments[0]
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
    >>> from skcomplex.core.base import symbol
    >>> from skcomplex.core.model import where
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
    >>> from skcomplex.core.base import symbol
    >>> from skcomplex.core.model import either, where
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
    >>> from skcomplex.core.base import symbol
    >>> from skcomplex.core.model import both, where
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
    >>> from skcomplex.core.base import symbol
    >>> from skcomplex.core.model import mod
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
    >>> from skcomplex.core.base import symbol
    >>> from skcomplex.core.model import derivative
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
    >>> from skcomplex.core.base import symbol
    >>> from skcomplex.core.model import lag
    >>> y = symbol('y')
    >>> lag(y, 1) - lag(y, 0)
    -y + lag(y, 1)
"""

delayed = _Delayed()
delayed.__doc__ = """Return the value of a state a delay before the current time, for delay differential equations.

The compiler replaces the delayed value with a value the solver takes from the run's history, and before the start
from the history declared on the system. A delay that depends on states needs its smallest value as `minimum`, which
bounds the steps of the method of steps.

Args:
    value:
        The symbol of a state.

    tau:
        The delay, a number, a parameter, or an expression.

    minimum:
        The smallest value the delay takes, needed when it is not a number or a parameter.

Returns:
    delayed:
        The symbolic delayed value.

Examples:
    >>> from skcomplex.core.base import symbol
    >>> from skcomplex.core.model import delayed
    >>> x, tau = symbol('x'), symbol('tau')
    >>> delayed(x, tau)
    delayed(x, tau)
"""


def step(t: Expression, at: float, height: float = 1.0) -> Expression:
    """Return a step input, 0 before a time and `height` from it on.

    Args:
        t:
            The time symbol, `s.t` in laws.

        at:
            The time of the step.

        height:
            The height of the step.

    Returns:
        input:
            The symbolic input.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import step
        >>> t = symbol('t')
        >>> [float(step(t, 5, 2).subs({t: time})) for time in (4, 5)]
        [0.0, 2.0]
    """
    return where(t >= at, height, 0)


def ramp(t: Expression, start: float, end: float, height: float = 1.0) -> Expression:
    """Return a ramp input, 0 before `start`, rising linearly to `height` at `end`, and constant after.

    Args:
        t:
            The time symbol, `s.t` in laws.

        start:
            The time the ramp starts.

        end:
            The time the ramp ends.

        height:
            The value after the ramp.

    Returns:
        input:
            The symbolic input.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import ramp
        >>> t = symbol('t')
        >>> [float(ramp(t, 0, 10, 5).subs({t: time})) for time in (-1, 4, 20)]
        [0.0, 2.0, 5.0]
    """
    return where(t < start, 0, where(t < end, height * (t - start) / (end - start), height))


def pulse(t: Expression, start: float, width: float, height: float = 1.0) -> Expression:
    """Return a rectangular pulse of a width and a height that starts at a time.

    Args:
        t:
            The time symbol, `s.t` in laws.

        start:
            The time the pulse starts.

        width:
            The duration of the pulse.

        height:
            The height of the pulse.

    Returns:
        input:
            The symbolic input.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import pulse
        >>> t = symbol('t')
        >>> [float(pulse(t, 2, 1, 3).subs({t: time})) for time in (1.5, 2.5, 3.5)]
        [0.0, 3.0, 0.0]
    """
    return where(both(t >= start, t < start + width), height, 0)


def pulse_train(t: Expression, period: float, width: float, height: float = 1.0, start: float = 0.0) -> Expression:
    """Return a periodic train of rectangular pulses, each of a width, starting at `start`.

    Args:
        t:
            The time symbol, `s.t` in laws.

        period:
            The time between the starts of pulses.

        width:
            The duration of every pulse.

        height:
            The height of every pulse.

        start:
            The time of the first pulse.

    Returns:
        input:
            The symbolic input.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import pulse_train
        >>> t = symbol('t')
        >>> [float(pulse_train(t, 10, 1).subs({t: time})) for time in (0.5, 5, 20.5)]
        [1.0, 0.0, 1.0]
    """
    return where(both(t >= start, mod(t - start, period) < width), height, 0)


def series(t: Expression, times: Sequence[float], values: Sequence[float], kind: str = 'linear') -> Expression:
    """Return an input that follows data, interpolated linearly or held from each time to the next.

    Before the first time the input takes the first value, and after the last time the last value.

    Args:
        t:
            The time symbol, `s.t` in laws.

        times:
            The increasing times of the data.

        values:
            The values at those times.

        kind:
            `'linear'` for linear interpolation, or `'previous'` to hold every value until the next time.

    Returns:
        input:
            The symbolic input, nested conditions that every backend lowers.

    Raises:
        ValueError:
            The times do not increase, the lengths differ, or the kind is unknown.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import series
        >>> t = symbol('t')
        >>> data = series(t, [0, 10, 20], [0.0, 5.0, 1.0])
        >>> [float(data.subs({t: time})) for time in (-5, 5, 15, 30)]
        [0.0, 2.5, 3.0, 1.0]
    """
    points, levels = [float(item) for item in times], [float(item) for item in values]
    if len(points) != len(levels) or not points or any(b <= a for a, b in itertools.pairwise(points)):
        error_msg = 'A series takes increasing times and as many values.'
        raise ValueError(error_msg)
    if kind not in {'linear', 'previous'}:
        error_msg = f"Unknown series kind {kind!r}. Use 'linear' or 'previous'."
        raise ValueError(error_msg)
    expression: Expression = SR(levels[-1])
    for index in range(len(points) - 2, -1, -1):
        start, end = points[index], points[index + 1]
        if kind == 'linear':
            segment = levels[index] + (levels[index + 1] - levels[index]) * (t - start) / (end - start)
        else:
            segment = SR(levels[index])
        expression = where(t < end, segment, expression)
    return where(t < points[0], levels[0], expression)


__test__ = {
    'where': where.__doc__,
    'either': either.__doc__,
    'both': both.__doc__,
    'mod': mod.__doc__,
    'derivative': derivative.__doc__,
    'lag': lag.__doc__,
    'delayed': delayed.__doc__,
}
