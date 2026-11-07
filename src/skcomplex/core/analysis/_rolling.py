"""Measure statistics of series in rolling windows: the variance, the lag-one autocorrelation, and the skewness."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_choice, _read_real, _read_series
from .types import Source

KINDS = ('variance', 'autocorrelation', 'skewness')
MIN_WINDOW = 3

TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class RollingResult:
    """Rolling statistic of every series of a source and its trend.

    Args:
        values:
            The series, one per column.

        labels:
            The label of every series.

        statistic:
            The statistic of every series in every window, one row per series, `nan` before the first full window.

        trend:
            Kendall's tau of the statistic of every series against time, positive when it rises.
    """

    values: Array
    labels: list[str]
    statistic: Array
    trend: Array


def rolling(
    source: Source,
    *,
    kind: str = 'variance',
    variables: Sequence[str] | None = None,
    window: int | None = None,
    detrend: bool = False,
) -> RollingResult:
    """Return the statistic of every series of a source in rolling windows, of one kind, and its trend.

    Every window holds the `window` values that end at a sample, possibly with a linear trend removed first, and the
    statistic of the window is set at that sample. The trend of the statistic is Kendall's tau against time.

    - `'variance'`: the variance of the window.
    - `'autocorrelation'`: the lag-one autocorrelation of the window.
    - `'skewness'`: the skewness of the window.

    Args:
        source:
            The trajectory, or an array with one row per time and one column per series.

        kind:
            The statistic to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        window:
            The number of values in every rolling window.

        detrend:
            Whether to remove a linear trend in every window first.

    Returns:
        result:
            The series, the rolling statistic of every series, and its trend.

    Raises:
        AnalysisError:
            The kind is unknown, or the window is missing or does not fit the series.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import rolling
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(2000)
        >>> for t in range(1, 2000):
        ...     x[t] = (0.2 + 0.79 * t / 2000) * x[t - 1] + rng.standard_normal()
        >>> result = rolling(x, kind='autocorrelation', window=200, detrend=True)
        >>> bool(result.trend[0] > 0.5)
        True
    """
    _check_choice('rolling', 'kind', kind, KINDS)
    if window is None or window < MIN_WINDOW:
        error_msg = f'rolling has window={window!r}. Give a window of {MIN_WINDOW} or more.'
        raise AnalysisError(error_msg)
    series = _read_series(source, variables)
    statistic = np.asarray(
        [_rolling(column, window, _statistic, kind, detrend=detrend) for column in series.values.T],
    )
    trend = np.asarray([_trend(row) for row in statistic])
    return RollingResult(series.values, series.labels, statistic, trend)


def _statistic(segment: Array, kind: str) -> float:
    """Return the statistic of one kind of a window."""
    if kind == 'variance':
        return float(segment.var())
    if kind == 'autocorrelation':
        return float(np.corrcoef(segment[:-1], segment[1:])[0, 1])
    return float(stats.skew(segment))


def _trend(statistic: Array) -> float:
    """Return Kendall's tau of a statistic against time over its defined values."""
    valid = ~np.isnan(statistic)
    return float(stats.kendalltau(np.flatnonzero(valid), statistic[valid]).statistic)


def _rolling(
    values: npt.ArrayLike,
    window: int,
    statistic: Callable[[Array, str], float],
    kind: str,
    *,
    detrend: bool,
) -> Array:
    """Return a statistic of every window of a series, set at the last sample of the window.

    Args:
        values:
            The series.

        window:
            The number of values in every window.

        statistic:
            The function of a window and of `kind` that returns the statistic.

        kind:
            The kind passed to the statistic.

        detrend:
            Whether to remove a linear trend in every window first.

    Returns:
        rolling:
            The statistic at every sample, `nan` before the first full window.
    """
    array = _read_real(values, 'values')
    if not MIN_WINDOW <= window <= array.size:
        error_msg = (
            f'window = {window} must lie between {MIN_WINDOW} and the length of the series, {array.size}. '
            'Use a window within that range.'
        )
        raise AnalysisError(error_msg)
    rolling = np.full(array.size, np.nan)
    axis = np.arange(window)
    for end in range(window, array.size + 1):
        segment = array[end - window : end]
        if detrend:
            segment = segment - np.polyval(np.polyfit(axis, segment, 1), axis)
        rolling[end - 1] = statistic(segment, kind)
    return rolling
