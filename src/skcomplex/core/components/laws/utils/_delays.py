"""Define the distributed delays of a value, expanded into hidden stages."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re

from sage.all__sagemath_symbolics import SR, Expression

from ._operators import _OperatorFunction

DELAY_NAME = re.compile(r'^(?P<kind>material|smoothing)_delay(?P<order>[1-9][0-9]*)$')
DELAY_FUNCTIONS: dict[str, _OperatorFunction] = {}


class _DelayFunction(_OperatorFunction):
    """Distributed delay of an input, named after its kind and its order."""

    def _eval_(self, value: Expression, time: Expression) -> Expression | None:
        """Return a constant input unchanged, or `None` otherwise.

        Every delay of a constant is the constant.
        """
        del time
        return SR(value) if SR(value).is_numeric() else None


def delay_parts(name: str) -> tuple[bool, int] | None:
    """Return whether a delay passes its input on whole, and its number of stages.

    It reads both from the name of the function that `distributed_delay` builds, such as `(True, 3)` for
    `material_delay3`, and gives `None` for a function that is no delay. The compiler reads it to add the hidden stocks
    of the delay. It finds the delays of a supply chain, a pipeline of trainees, and a smoothed forecast.

    Read more in the user guide, under Extend the library.

    Args:
        name:
            The name of a function.

    Returns:
        kind:
            Whether the delay passes its input on whole, and its number of stages, or `None`.

    See Also:
        distributed_delay: Builds the delays it reads.

    Examples:
        >>> from skcomplex.core.components.laws.utils._delays import delay_parts
        >>> delay_parts('material_delay3'), delay_parts('sin')
        ((True, 3), None)
    """
    match = DELAY_NAME.match(name)
    return None if match is None else (match['kind'] == 'material', int(match['order']))


def distributed_delay(value: object, time: object, order: int = 1, *, conserved: bool = True) -> Expression:
    """Return a value delayed through a chain of hidden stocks.

    The delay is `order` hidden stocks in a row, each holding the value for `time / order` on average. A conserved delay
    passes its input on like goods in transit, so its output is the outflow of the last stock and a pulse comes out
    spread over time with the mean `time`. A delay that is not conserved smooths its input, each stock moving toward the
    one before it at `gap / (time / order)`, such as `distributed_delay(sales, 2, conserved=False)` for a smoothed
    forecast of sales. These are the `DELAY1`, `DELAY3`, and `SMOOTH` functions of system dynamics. It delays orders in
    a supply chain, trainees in a hiring pipeline, and the maturation of cells.

    Read more in the user guide, under Laws.

    Args:
        value:
            The symbolic input.

        time:
            The mean delay time, a positive number or an expression of parameters.

        order:
            The number of hidden stocks.

        conserved:
            Whether the delay passes its input on whole, as goods in transit, or smooths it, as a forecast.

    Returns:
        output:
            The symbolic output.

    Raises:
        ValueError:
            The order is not a positive whole number, or the delay time is a number that is not positive.

    See Also:
        fractional: Gives a state a memory of its whole past.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import distributed_delay
        >>> distributed_delay(var('orders'), 4, order=3)
        material_delay3(orders, 4)
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        error_msg = f'The order of a distributed delay must be a positive whole number, but it is {order!r}.'
        raise ValueError(error_msg)
    name = f'{"material" if conserved else "smoothing"}_delay{order}'
    if SR(time).is_numeric() and float(SR(time)) <= 0:
        error_msg = f'The delay time of a distributed delay must be positive, but it is {time}.'
        raise ValueError(error_msg)
    if name not in DELAY_FUNCTIONS:
        DELAY_FUNCTIONS[name] = _DelayFunction(name, nargs=2)
    return DELAY_FUNCTIONS[name](value, time)
