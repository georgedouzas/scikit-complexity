"""Define the signals of time that laws drive states with.

They are steps, ramps, pulses, and series.
"""

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
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import step
        >>> t = var('t')
        >>> [float(step(t, 5, 2).subs({t: time})) for time in (4, 5)]
        [0.0, 2.0]
    """
    return where(t >= at, height, 0)


def ramp(t: Expression, start: float, end: float, height: float = 1.0) -> Expression:
    """Return a ramp input.

    It is 0 before `start`, rises linearly to `height` at `end`, and is constant after.

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
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import ramp
        >>> t = var('t')
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
    """Return a periodic train of rectangular pulses.

    Each pulse has a width, and the train starts at `start`.

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
            The count is not a positive integer.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import pulse_train
        >>> t = var('t')
        >>> [float(pulse_train(t, 10, 1).subs({t: time})) for time in (0.5, 5, 20.5)]
        [1.0, 0.0, 1.0]
        >>> [float(pulse_train(t, 10, 1, count=2).subs({t: time})) for time in (10.5, 20.5)]
        [1.0, 0.0]
    """
    if count is None:
        return where(both(t >= start, mod(t - start, period) < width), height, 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        error_msg = f'A pulse train takes a positive integer count of pulses or None, but got {count!r}.'
        raise ValidationError(error_msg)
    return where(both(t >= start, t < start + count * period, mod(t - start, period) < width), height, 0)


def periodic_guard(t: Expression, period: float, start: float = 0.0, count: int | None = None) -> Expression:
    """Return a guard that crosses zero at periodic times.

    The guard crosses zero once at every `start + j * period` for `j` from 0, `count` times or without end, and stays
    away from zero between those times and after the last one. A crossing law with this guard and the direction 0 fires
    at every such time.

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
            The period is not positive, or the count is not a positive integer.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import periodic_guard
        >>> t = var('t')
        >>> guard = periodic_guard(t, 2.0, count=2)
        >>> [round(float(guard.subs({t: time})), 6) for time in (0.0, 1.0, 2.0, 4.0)]
        [0.0, 1.0, 0.0, -1.0]
    """
    if not period > 0:
        error_msg = f'A periodic guard takes a positive period, but got {period!r}.'
        raise ValidationError(error_msg)
    guard = where(t < start - period / 2, -1, sin(pi * (t - start) / period))
    if count is None:
        return guard
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        error_msg = f'A periodic guard takes a positive integer count of crossings or None, but got {count!r}.'
        raise ValidationError(error_msg)
    last = count - SR(1) / 2
    return where(t > start + last * period, sin(pi * last), guard)


def series(t: Expression, times: Sequence[float], values: Sequence[float], kind: str = 'linear') -> Expression:
    """Return an input that follows data.

    It is interpolated linearly or held from each time to the next. Before the first time the input takes the first
    value, and after the last time the last value.

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
            The symbolic input, a balanced search of conditions on time that every backend lowers.

    Raises:
        ValueError:
            The times do not increase, the lengths differ, or the kind is unknown.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import series
        >>> t = var('t')
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
    return where(t < points[0], levels[0], _search(t, points, levels, kind, (0, len(points) - 1)))
