"""Define the signals of time that drive states, such as steps, ramps, and pulses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression, pi, sin

from ....errors import ValidationError
from ._conditions import both, mod, where


def _search(t: Expression, points: list[float], levels: list[float], kind: str, span: tuple[int, int]) -> Expression:
    """Return the pieces of a series in a span of indices as a binary search.

    The search is a balanced binary search of conditions on time.
    """
    low, high = span
    if low == high:
        if kind == 'previous' or low == len(points) - 1:
            return SR(levels[low])
        start, end = points[low], points[low + 1]
        return levels[low] + (levels[low + 1] - levels[low]) * (t - start) / (end - start)
    middle = (low + high + 1) // 2
    return where(
        t < points[middle],
        _search(t, points, levels, kind, (low, middle - 1)),
        _search(t, points, levels, kind, (middle, high)),
    )


def step(t: Expression, at: float, height: float = 1.0) -> Expression:
    """Return a step input, 0 before a time and `height` from it on.

    It switches a law on at the time `at` and keeps it on, such as `step(t, 5, 2)` that is 0 at time 4 and 2 at time 5.
    It opens a valve, raises a tax rate, and starts a dose of a drug.

    Read more in the user guide, under Laws.

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

    See Also:
        ramp: Rises from 0 to a height over an interval.
        pulse: Is on for an interval only.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import step
        >>> t = var('t')
        >>> [float(step(t, 5, 2).subs({t: time})) for time in (4, 5)]
        [0.0, 2.0]
    """
    return where(t >= at, height, 0)


def ramp(t: Expression, start: float, end: float, height: float = 1.0) -> Expression:
    """Return an input that rises in a straight line from 0 to a height.

    It is 0 before `start`, rises evenly to `height` at `end`, and stays there, such as `ramp(t, 0, 10, 5)` that is 2 at
    time 4. It phases in a subsidy, warms an oven to its setting, and raises the load on a bridge.

    Read more in the user guide, under Laws.

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

    See Also:
        step: Switches from 0 to a height at once.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import ramp
        >>> t = var('t')
        >>> [float(ramp(t, 0, 10, 5).subs({t: time})) for time in (-1, 4, 20)]
        [0.0, 2.0, 5.0]
    """
    return where(t < start, 0, where(t < end, height * (t - start) / (end - start), height))


def pulse(t: Expression, start: float, width: float, height: float = 1.0) -> Expression:
    """Return a rectangular pulse of a width and a height that starts at a time.

    It is `height` from `start` until `start + width` and 0 otherwise, such as `pulse(t, 2, 1, 3)` that is 3 at time 2.5
    and 0 at time 3.5. It gives a single dose of a drug, a short shock to demand, and a burst of current into a neuron.

    Read more in the user guide, under Laws.

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

    See Also:
        pulse_train: Repeats a pulse at a period.
        step: Switches from 0 to a height and stays there.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import pulse
        >>> t = var('t')
        >>> [float(pulse(t, 2, 1, 3).subs({t: time})) for time in (1.5, 2.5, 3.5)]
        [0.0, 3.0, 0.0]
    """
    return where(both(t >= start, t < start + width), height, 0)


def pulse_train(
    t: Expression,
    period: float,
    width: float,
    height: float = 1.0,
    start: float = 0.0,
    *,
    count: int | None = None,
) -> Expression:
    """Return a train of rectangular pulses repeated at a period.

    A pulse of `height` lasting `width` starts at `start` and again every `period`, without end or `count` times, such
    as `pulse_train(t, 10, 1)` that is 1 at times 0.5 and 20.5 and 0 at time 5. It gives daily doses of a drug, the
    beats of a pacemaker, and the monthly payments of a loan.

    Read more in the user guide, under Laws.

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

        count:
            The number of pulses, or `None` for a train that never ends.

    Returns:
        input:
            The symbolic input.

    Raises:
        ValidationError:
            The count is not a positive whole number.

    See Also:
        pulse: Gives a single pulse.
        periodic_guard: Crosses zero at periodic times, for crossing laws.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import pulse_train
        >>> t = var('t')
        >>> [float(pulse_train(t, 10, 1).subs({t: time})) for time in (0.5, 5, 20.5)]
        [1.0, 0.0, 1.0]
    """
    if count is None:
        return where(both(t >= start, mod(t - start, period) < width), height, 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        error_msg = f'The count of a pulse train must be a positive whole number or None, but it is {count!r}.'
        raise ValidationError(error_msg)
    return where(both(t >= start, t < start + count * period, mod(t - start, period) < width), height, 0)


def periodic_guard(t: Expression, period: float, start: float = 0.0, count: int | None = None) -> Expression:
    """Return an expression of time that crosses zero at periodic times.

    It crosses zero once at every `start + j * period` for `j` from 0, `count` times or without end, and stays away from
    zero between those times and after the last one, such as `periodic_guard(t, 2.0, count=2)` that is 0 at times 0 and
    2 only. A `CrossingLaw` with this guard and the direction 0 acts at every such time. It pays a dividend every
    quarter, harvests a crop every year, and gives a vaccine at fixed dates.

    Read more in the user guide, under Laws.

    Args:
        t:
            The time symbol, `s.t` in laws.

        period:
            The time between two crossings.

        start:
            The time of the first crossing.

        count:
            The number of crossings, or `None` for crossings without end.

    Returns:
        guard:
            The symbolic guard.

    Raises:
        ValidationError:
            The period is not positive, or the count is not a positive whole number.

    See Also:
        CrossingLaw: Acts where the guard crosses zero.
        pulse_train: Repeats a pulse at a period, for rates.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import periodic_guard
        >>> t = var('t')
        >>> guard = periodic_guard(t, 2.0, count=2)
        >>> [round(float(guard.subs({t: time})), 6) for time in (0.0, 1.0, 2.0, 4.0)]
        [0.0, 1.0, 0.0, -1.0]
    """
    if not period > 0:
        error_msg = f'The period of a periodic guard must be positive, but it is {period!r}.'
        raise ValidationError(error_msg)
    guard = where(t < start - period / 2, -1, sin(pi * (t - start) / period))
    if count is None:
        return guard
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        error_msg = f'The count of a periodic guard must be a positive whole number or None, but it is {count!r}.'
        raise ValidationError(error_msg)
    last = count - SR(1) / 2
    return where(t > start + last * period, sin(pi * last), guard)


def series(x: object, points: Sequence[float], values: Sequence[float], kind: str = 'linear') -> Expression:
    """Return a curve through given points, joined by lines or held as steps.

    Between two points it draws a straight line, or holds the earlier value with `kind='previous'`, and it keeps the
    first value before the first point and the last value after the last point, such as 2.5 at time 5 for
    `series(t, [0, 10, 20], [0.0, 5.0, 1.0])`. With the time `s.t` as the value it is an input that follows data, and
    with any other expression it is a lookup table, such as the table functions of system dynamics. It replays measured
    rainfall, maps food to fertility, and gives a tax rate by income.

    Read more in the user guide, under Laws.

    Args:
        x:
            The symbolic value the series reads, such as the time symbol `s.t` or a state.

        points:
            The increasing points of the data.

        values:
            The values at those points.

        kind:
            `'linear'` for linear interpolation, or `'previous'` to hold every value until the next point.

    Returns:
        series:
            The symbolic curve, nested choices on the value that every backend can run.

    Raises:
        ValueError:
            The points do not increase, the lengths differ, or the kind is unknown.

    See Also:
        ramp: Rises in a straight line between two times.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import series
        >>> t = var('t')
        >>> data = series(t, [0, 10, 20], [0.0, 5.0, 1.0])
        >>> [float(data.subs({t: time})) for time in (-5, 5, 15, 30)]
        [0.0, 2.5, 3.0, 1.0]
    """
    value = SR(x)
    levels = [float(item) for item in values]
    points = [float(item) for item in points]
    if len(points) != len(levels) or not points or any(b <= a for a, b in itertools.pairwise(points)):
        error_msg = 'The points of a series must increase, and every point needs one value.'
        raise ValueError(error_msg)
    if kind not in {'linear', 'previous'}:
        error_msg = f"Unknown series kind {kind!r}. Use 'linear' or 'previous'."
        raise ValueError(error_msg)
    return where(value < points[0], levels[0], _search(value, points, levels, kind, (0, len(points) - 1)))
