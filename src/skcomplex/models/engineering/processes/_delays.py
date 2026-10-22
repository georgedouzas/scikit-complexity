"""System-dynamics delays and smoothings, realized as distributed delays of the core."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Expression
from ....core.laws import distributed_delay


def delay1(value: object, time: object) -> Expression:
    """Return the outflow of a first-order material delay of an inflow, one hidden stock emptied at `stock / time`.

    Args:
        value:
            The symbolic inflow.

        time:
            The average delay time, a number or an expression of parameters.

    Returns:
        output:
            The symbolic outflow.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.models.engineering.processes import delay1
        >>> delay1(symbol('orders'), 4)
        material_delay1(orders, 4)
    """
    return distributed_delay(value, time, 1)


def delay3(value: object, time: object) -> Expression:
    """Return the outflow of a third-order material delay, three hidden stocks in series each with a third of the time.

    The response to a step is the Erlang distribution of order 3 with the mean `time`.

    Args:
        value:
            The symbolic inflow.

        time:
            The average delay time, a number or an expression of parameters.

    Returns:
        output:
            The symbolic outflow.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.models.engineering.processes import delay3
        >>> delay3(symbol('shipments'), 6)
        material_delay3(shipments, 6)
    """
    return distributed_delay(value, time, 3)


def smooth(value: object, time: object) -> Expression:
    """Return the first-order information smoothing of a value, a hidden stock that adjusts at `gap / time`.

    Args:
        value:
            The symbolic value.

        time:
            The smoothing time, a number or an expression of parameters.

    Returns:
        output:
            The symbolic smoothed value, which starts at the value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.models.engineering.processes import smooth
        >>> smooth(symbol('sales'), 2)
        smoothing_delay1(sales, 2)
    """
    return distributed_delay(value, time, 1, conserved=False)


def smooth3(value: object, time: object) -> Expression:
    """Return the third-order information smoothing of a value, three smoothings each with a third of the time.

    Args:
        value:
            The symbolic value.

        time:
            The smoothing time, a number or an expression of parameters.

    Returns:
        output:
            The symbolic smoothed value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.models.engineering.processes import smooth3
        >>> smooth3(symbol('sales'), 3)
        smoothing_delay3(sales, 3)
    """
    return distributed_delay(value, time, 3, conserved=False)
