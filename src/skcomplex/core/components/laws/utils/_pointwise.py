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
    """Return the Hill function `x^n / (half^n + x^n)`, which rises from 0 to 1.

    It is one half at `x = half`, such as `hill(x, 2, 3)` that gives `x^3 / (x^3 + 8)`, and a larger `n` makes the rise
    steeper, close to a switch. The order `n = 1` gives the saturating function `x / (half + x)`, and
    `1 - hill(x, half, n)` falls from 1 to 0. It gives the activation of a gene by a protein, the uptake of a nutrient,
    and the response of a predator to more prey.

    Read more in the user guide, under Laws.

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

    See Also:
        logistic: Rises from 0 to 1 through a center for any real input.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import hill
        >>> hill(var('x'), 2, 3)
        x^3/(x^3 + 8)
    """
    return SR(x) ** n / (SR(half) ** n + SR(x) ** n)


def logistic(x: object, center: object = 0, width: object = 1) -> Expression:
    """Return the logistic curve `1 / (1 + exp(-(x - center) / width))`.

    It rises from 0 to 1 and is one half at `x = center`, such as `logistic(x, 1, 2)` that is one half at 1. It is
    written as `(1 + tanh((x - center) / (2 width))) / 2`, which gives the same value and stays accurate for large
    inputs, where the exponential form that Sage may produce loses all its digits. It gives the firing rate of a neuron,
    the chance that a buyer adopts a product, and the share of a population that has caught on.

    Read more in the user guide, under Laws.

    Args:
        x:
            The expression the function reads.

        center:
            The value of `x` where the function is one half.

        width:
            The scale of the rise, the inverse of its steepness, negative for a falling curve.

    Returns:
        value:
            The symbolic value.

    See Also:
        hill: Rises from 0 to 1 for a nonnegative input.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import logistic
        >>> logistic(var('x'), 1, 2)
        1/2*tanh(1/4*x - 1/4) + 1/2
    """
    return (1 + tanh((SR(x) - center) / (2 * SR(width)))) / 2


def clip(x: object, low: object = None, high: object = None) -> Expression:
    """Return an expression limited to an interval.

    It gives `low` where the expression is below it, `high` where it is above it, and the expression itself in between,
    such as 0, 0.5, and 1 for `clip(x, 0, 1)` at -2, 0.5, and 3. It keeps a concentration from turning negative, an
    order within the stock a firm holds, and a control signal within what a motor can do.

    Read more in the user guide, under Laws.

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

    See Also:
        sign: Gives the sign of an expression.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import clip
        >>> x = var('x')
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

    It gives the direction of a value without its size, such as -1, 0, and 1 for `sign(x)` at -2, 0, and 3, and its
    derivative is 0 away from 0, so solvers that need derivatives stay finite. It turns a field into the spin of a
    magnet, a signal into the side a trader takes, and a velocity into the direction of friction.

    Read more in the user guide, under Laws.

    Args:
        x:
            The expression.

    Returns:
        value:
            The symbolic sign.

    See Also:
        clip: Limits an expression to an interval.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import sign
        >>> x = var('x')
        >>> [sign(x).subs(x=value) for value in (-2, 0, 3)]
        [-1, 0, 1]
    """
    return where(SR(x) > 0, 1, where(SR(x) < 0, -1, 0))


def stop_gradient(x: object) -> Expression:
    """Return an expression unchanged, but with a gradient of zero.

    The value is the expression itself, and a gradient taken through it is 0, so `x * stop_gradient(x)` differentiates
    to `stop_gradient(x)`. The JAX backend calls `jax.lax.stop_gradient` and every other backend gives the value as it
    is. The sum `exact + relaxed - stop_gradient(relaxed)` has the value of `exact` and the gradient of `relaxed`. It
    trains a network of spiking neurons, rounds the bids of a learned market, and fixes a target in reinforcement
    learning.

    Read more in the user guide, under Laws.

    Args:
        x:
            The expression the function reads.

    Returns:
        value:
            The symbolic value.

    See Also:
        sign: Gives the sign of an expression, a common exact value.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import stop_gradient
        >>> x = var('x')
        >>> (x * stop_gradient(x)).diff(x)
        stop_gradient(x)
    """
    return STOP_GRADIENT(SR(x))
