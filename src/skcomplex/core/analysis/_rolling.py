"""Measure statistics of series in rolling windows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..enums import DataKind
from ..errors import AnalysisError
from ..types import Array
from ._analysis_tags import AnalysisTags
from ._base import _check_choice, _read_real, _read_series
from .types import DataSource

KINDS = ('variance', 'autocorrelation', 'skewness')
MIN_WINDOW = 3
TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class RollingResult:
    """A result of `rolling`, a statistic in rolling windows and its trend.

    It holds the series, the statistic of every window set at its last sample, and the trend of the statistic, such as
    a trend of 1 for a variance that grows in every window. It reports the early warnings of a lake that turns turbid,
    a market that crashes, or a climate that shifts.

    Read more in the user guide, under Analyse.

    Args:
        values:
            The series, one per column.

        labels:
            The label of every series.

        statistic:
            The statistic of every series in every window, one row per series, `nan` before the first full window.

        trend:
            Kendall's tau of the statistic of every series against time, positive when it rises.

    See Also:
        rolling: Returns this result.

    Examples:
        >>> from skcomplex.core.analysis import rolling
        >>> rolling([0, 1, 0, 2, 0, 3], window=3).statistic.round(2).tolist()
        [[nan, nan, 0.22, 0.67, 0.89, 1.56]]
    """

    values: Array
    labels: list[str]
    statistic: Array
    trend: Array


def rolling(
    source: DataSource,
    *,
    kind: str = 'variance',
    variables: Sequence[str] | None = None,
    window: int | None = None,
    detrend: bool = False,
) -> RollingResult:
    """Return the statistic of every series in rolling windows, and its trend.

    Every window holds the `window` values that end at a sample, possibly with a linear trend removed first, and the
    statistic of the window is set at that sample. The trend of the statistic is the tau of Kendall against time, such
    as 1 for the variance of `[0, 1, 0, 2, 0, 3]`, whose swings grow. It finds the early warnings of a tipping point,
    such as a rising variance before a lake turns turbid, a market crashes, or a climate shifts.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The trajectory, or an array with one row per time and one column per series.

        kind:
            What to return. `'variance'` gives the variance of every window, `'autocorrelation'` the correlation of
            every window with itself one step later, and `'skewness'` the skewness of every window.

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

    See Also:
        correlation: Gives the correlation of series, of an ensemble, or of a field.
        events: Finds the runs of a series above a threshold.

    Examples:
        >>> from skcomplex.core.analysis import rolling
        >>> rolling([0, 1, 0, 2, 0, 3], window=3).trend.tolist()
        [1.0]
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
    """Return a statistic of every window of a series.

    It is set at the last sample of the window.

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
