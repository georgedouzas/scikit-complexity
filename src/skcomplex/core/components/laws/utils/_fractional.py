"""Define the Caputo derivative of a fractional order that rate laws target."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression
from sage.symbolic.function import SymbolicFunction

from ....errors import ValidationError
from ._operators import _OperatorFunction


class _FractionalFunction(_OperatorFunction):
    """Caputo derivative of a state, printed with its order as a short number."""

    def _print_(self, state: Expression, order: Expression) -> str:
        """Return the derivative call with the order as Python prints it."""
        return f'fractional({state}, {float(order)!r})'


FRACTIONAL = _FractionalFunction('fractional', nargs=2)


def fractional(x: object, alpha: float) -> Expression:
    """Return the derivative of a fractional order of a value, for a rate law.

    The Caputo derivative `D^alpha x` of an order `0 < alpha < 1` weighs the whole past of the value, so a value it
    moves remembers its history. `RateLaw(fractional(x, 0.5), -k * x)` states `D^0.5 x = -k x`, a decay slower than any
    exponential, as the Mittag-Leffler function `E_alpha(-k t^alpha)`, and the order 1 is the ordinary first derivative.
    It models charges that relax in a dielectric, stress in a material that flows slowly, and drugs held in tissue.

    Read more in the user guide, under Laws.

    Args:
        x:
            The state, as the interaction reads it, such as `s.target.x`.

        alpha:
            The order of the derivative, a number in `(0, 1]`.

    Returns:
        derivative:
            The unevaluated derivative, the target of a `RateLaw`.

    Raises:
        ValidationError:
            The order is not a number in `(0, 1]`, or `x` is not a state.

    See Also:
        RateLaw: Takes the derivative as its target.
        distributed_delay: Delays a value through hidden stocks.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.components.laws.utils import fractional
        >>> x, k = var('x k')
        >>> RateLaw(fractional(x, 0.5), -k * x).target
        fractional(x, 0.5)
    """
    order = SR(alpha)
    if not order.is_numeric() or not order.is_real() or not 0 < float(order) <= 1:
        error_msg = (
            f'The order of a fractional derivative is a number in (0, 1], but got {alpha!r}. Give an order such as '
            '`fractional(x, 0.5)`.'
        )
        raise ValidationError(error_msg)
    state = SR(x)
    applied = isinstance(state.operator(), SymbolicFunction) and len(state.operands()) == 1
    if not state.is_symbol() and not applied:
        error_msg = (
            f'A fractional derivative is taken of a state, such as `fractional(s.target.x, 0.5)`, but got {x}. '
            'Move the other terms to the rate.'
        )
        raise ValidationError(error_msg)
    return FRACTIONAL(state, order)


def fractional_of(target: object) -> tuple[Expression, float] | None:
    """Return the value and the order of a fractional derivative, or `None`.

    It reads back what `fractional` built, such as `(x, 0.5)` for `fractional(x, 0.5)`, and gives `None` for any other
    target, so the compiler picks the Caputo solver for the laws that need it. It finds the fractional decays of
    dielectrics, slowly flowing materials, and drugs held in tissue.

    Read more in the user guide, under Extend the library.

    Args:
        target:
            The target of a law.

    Returns:
        derivative:
            The value and the order as a float, or `None` when the target is no fractional derivative.

    See Also:
        fractional: Builds the derivatives it reads.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import fractional
        >>> from skcomplex.core.components.laws.utils._fractional import fractional_of
        >>> fractional_of(fractional(var('x'), 0.5))
        (x, 0.5)
    """
    if not isinstance(target, Expression) or target.operator() != FRACTIONAL:
        return None
    state, order = target.operands()
    return state, float(order)
