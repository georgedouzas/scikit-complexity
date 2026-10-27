"""Define the distributed delays of system dynamics, as Sage symbolic functions the compiler expands into stages.

Sage has no delay of an input through hidden stages, so every delay is a Sage function made by `function`, named after
its kind and its order and kept unevaluated in expressions. The compiler adds its stages as hidden stocks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re

from sage.all__sagemath_symbolics import SR, Expression, function

DELAY_NAME = re.compile(r'^(?P<kind>material|smoothing)_delay(?P<order>[1-9][0-9]*)$')


def _eval_delay(self: object, value: Expression, time: Expression) -> Expression | None:
    """Return a constant input unchanged, since every delay of a constant is the constant, or `None` otherwise."""
    del self, time
    return SR(value) if SR(value).is_numeric() else None


def delay_parts(name: str) -> tuple[bool, int] | None:
    """Return whether a delay function conserves its input, and its order, from its name, or `None` for no delay.

    Args:
        name:
            The name of a function.

    Returns:
        parts:
            Whether the delay is material, and its order.

    Examples:
        >>> from skcomplex.core.laws.utils import delay_parts
        >>> delay_parts('material_delay3'), delay_parts('sin')
        ((True, 3), None)
    """
    match = DELAY_NAME.match(name)
    return None if match is None else (match['kind'] == 'material', int(match['order']))


def distributed_delay(value: object, time: object, order: int = 1, *, conserved: bool = True) -> Expression:
    """Return the output of a distributed delay of a value: hidden stages in series, each with a share of the time.

    A conserved delay passes its input through the stages like a material, so its output is the outflow of the last
    stage, `stage / (time / order)`, and its response to a pulse is the Erlang distribution of the order with the mean
    `time`. A delay that does not conserve its input smooths it, each stage adjusting to the previous one at
    `gap / (time / order)`, and its output starts at the value.

    Args:
        value:
            The symbolic input.

        time:
            The mean delay time, a positive number or an expression of parameters.

        order:
            The number of stages.

        conserved:
            Whether the delay conserves its input, as a material delay, or smooths it, as an information delay.

    Returns:
        output:
            The symbolic output.

    Raises:
        ValueError:
            The order is not a positive integer, or the delay time is a number that is not positive.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import distributed_delay
        >>> distributed_delay(var('orders'), 4, order=3)
        material_delay3(orders, 4)
        >>> distributed_delay(var('sales'), 2, conserved=False)
        smoothing_delay1(sales, 2)
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        error_msg = f'A distributed delay has a positive integer order, but got {order!r}.'
        raise ValueError(error_msg)
    name = f'{"material" if conserved else "smoothing"}_delay{order}'
    if SR(time).is_numeric() and float(SR(time)) <= 0:
        error_msg = f'The delay time of {name} must be positive, but got {time}.'
        raise ValueError(error_msg)
    return function(name, nargs=2, eval_func=_eval_delay)(value, time)
