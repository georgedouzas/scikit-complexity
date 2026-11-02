"""Count the equal steps a solver takes over a span of time, in Python and in compiled loops."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from numba.extending import register_jitable

STEP_TOLERANCE = 1e-9


@register_jitable
def step_count(t: float, until: float, dt: float) -> int:
    """Return the number of equal steps no longer than `dt` from time `t` to time `until`, at least one.

    A span within `1e-9` steps above a whole number of steps takes that whole number, so rounding in the span does not
    add a step. The compiled loops of the Numba backend call the same function.

    Args:
        t:
            The start time.

        until:
            The end time.

        dt:
            The largest step.

    Returns:
        count:
            The number of steps.

    Examples:
        >>> from skcomplex.engine.problems._steps import step_count
        >>> step_count(0.1, 0.4, 0.1)
        3
    """
    return max(1, math.ceil((until - t) / dt - STEP_TOLERANCE))
