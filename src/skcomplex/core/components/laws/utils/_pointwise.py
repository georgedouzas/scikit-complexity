"""Define the pointwise functions of laws, such as the Hill and logistic functions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression, tanh
from sage.symbolic.function import BuiltinFunction

from ._conditions import where


class _StopGradientFunction(BuiltinFunction):
    """Identity whose gradient is zero, `stop_gradient(x)`."""

    def __init__(self) -> None:
        super().__init__('stop_gradient', nargs=1)

    def _eval_(self, x: Expression) -> Expression | None:
        """Return a number unchanged, or `None` to stay unevaluated."""
        value = SR(x)
        return value if value.is_numeric() else None

    def _tderivative_(self, x: Expression, diff_param: Expression) -> Expression:
        """Return the derivative, which is zero."""
        return SR(0)


STOP_GRADIENT = _StopGradientFunction()


def _smallest(values: tuple[object, ...]) -> Expression:
    """Return the smallest of several expressions, member by member, as nested choices.

    Args:
        values:
            The expressions, at least one.

    Returns:
        smallest:
            The symbolic smallest value.
    """
    result = SR(values[0])
    for value in values[1:]:
        result = where(SR(value) < result, value, result)
    return result


def _largest(values: tuple[object, ...]) -> Expression:
    """Return the largest of several expressions, member by member, as nested choices.

    Args:
        values:
            The expressions, at least one.

    Returns:
        largest:
            The symbolic largest value.
    """
    result = SR(values[0])
    for value in values[1:]:
        result = where(SR(value) > result, value, result)
    return result


def hill(x: object, half: object, n: object = 1) -> Expression:
    """Return the Hill function `x^n / (half^n + x^n)`.

    It rises from 0 to 1 and is one half at `x = half`. The order `n = 1` gives the saturating function `x / (half +
    x)`, and `1 - hill(x, half, n)` is the decreasing Hill function.

    Args:
        x:
            The expression the function reads, nonnegative.

        half:
            The value of `x` where the function is one half.

        n:
            The Hill coefficient, the steepness of the rise.

    Returns:
        value:
            The symbolic value.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import hill
        >>> x = var('x')
        >>> hill(x, 2, 3)
        x^3/(x^3 + 8)
        >>> float(hill(x, 2, 3).subs(x=2))
        0.5
    """
    return SR(x) ** n / (SR(half) ** n + SR(x) ** n)


def logistic(x: object, center: object = 0, width: object = 1) -> Expression:
    """Return the logistic sigmoid `1 / (1 + exp(-(x - center) / width))`.

    It rises from 0 to 1 through `center`. It is written as `(1 + tanh((x - center) / (2 width))) / 2`, which stays
    accurate for every argument. Sage rewrites the exponential of some arguments, such as a multiple of the logarithm of
    a constant, as `cosh(u) - sinh(u)`, whose terms cancel for a large `u`, so the sigmoid would hold rounding noise of
    the size of 1, or divide by zero.

    Args:
        x:
            The expression the function reads.

        center:
            The value of `x` where the function is one half.

        width:
            The scale of the rise, the inverse of its gain, negative for a falling sigmoid.

    Returns:
        value:
            The symbolic value.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import logistic
        >>> x = var('x')
        >>> logistic(x, 1, 2)
        1/2*tanh(1/4*x - 1/4) + 1/2
        >>> float(logistic(x, 1, 2).subs(x=1))
        0.5
    """
    return (1 + tanh((SR(x) - center) / (2 * SR(width)))) / 2


def clip(x: object, low: object = None, high: object = None) -> Expression:
    """Return an expression limited to an interval.

    It is `low` below it, `high` above it, and the expression inside.

    Args:
        x:
            The expression to limit.

        low:
            The lower bound, or `None` for none.

        high:
            The upper bound, or `None` for none.

    Returns:
        value:
            The symbolic limited value.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import clip
        >>> x = var('x')
        >>> clip(x, 0, 1)
        where(x < 0, 0, where(x > 1, 1, x))
        >>> [clip(x, 0, 1).subs(x=value) for value in (-2, 0.5, 3)]
        [0, 0.5, 1]
    """
    value = SR(x)
    if high is not None:
        value = where(SR(x) > high, high, value)
    if low is not None:
        value = where(SR(x) < low, low, value)
    return value


def sign(x: object) -> Expression:
    """Return the sign of an expression, 1 above 0, -1 below 0, and 0 at 0.

    Args:
        x:
            The expression.

    Returns:
        value:
            The symbolic sign.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import sign
        >>> x = var('x')
        >>> sign(x)
        where(x > 0, 1, where(x < 0, -1, 0))
        >>> [sign(x).subs(x=value) for value in (-2, 0, 3)]
        [-1, 0, 1]
    """
    return where(SR(x) > 0, 1, where(SR(x) < 0, -1, 0))


def stop_gradient(x: object) -> Expression:
    """Return a value whose gradient is zero.

    The JAX backend lowers it to `jax.lax.stop_gradient` and every other backend to the identity, so it changes
    gradients and never values. The straight-through estimator `exact + relaxed - stop_gradient(relaxed)` has the value
    `exact` and the gradient of `relaxed`.

    Args:
        x:
            The expression the function reads.

    Returns:
        value:
            The symbolic value.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import stop_gradient
        >>> x = var('x')
        >>> stop_gradient(x**2)
        stop_gradient(x^2)
        >>> (x * stop_gradient(x)).diff(x)
        stop_gradient(x)
    """
    return STOP_GRADIENT(SR(x))
