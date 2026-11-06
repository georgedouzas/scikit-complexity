"""Define the derivatives, lags, and delayed values of states that laws read, which the compiler replaces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression
from sage.symbolic.function import BuiltinFunction

DELAY_ARGUMENTS = 2


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
        if len(arguments) >= DELAY_ARGUMENTS and SR(arguments[1]).is_numeric() and SR(arguments[1]).pyobject() == 0:
            return arguments[0]
        return None


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
    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws.utils import derivative
    >>> x = var('x')
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
    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws.utils import lag
    >>> y = var('y')
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
    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws.utils import delayed
    >>> x, tau = var('x'), var('tau')
    >>> delayed(x, tau)
    delayed(x, tau)
"""


class _Residence(BuiltinFunction):
    """Number of whole steps a state of a discrete map has held its current value, `residence(x)`."""

    def __init__(self) -> None:
        super().__init__('residence', nargs=1)


residence = _Residence()


residence.__doc__ = """Return the number of whole steps a state of a discrete map has held its current value.

The residence is 0 in the step in which the state takes a value, and grows by 1 every step it keeps it. The compiler
replaces it with hidden states that keep the previous value and the count of every member.

Args:
    value:
        The symbol of a state.

Returns:
    residence:
        The symbolic residence.

Examples:
    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws.utils import residence
    >>> x = var('x')
    >>> residence(x) >= 3
    residence(x) >= 3
"""
