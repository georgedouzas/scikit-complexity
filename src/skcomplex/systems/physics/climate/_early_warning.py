"""Measure early warnings of tipping: rolling variance, autocorrelation, skewness, and spectral reddening."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ....core.analysis import rolling
from ....core.analysis.types import Source
from ....core.types import Array


@dataclass(frozen=True)
class EarlyWarningResult:
    """Rolling indicators of critical slowing down of every series, and their trends.

    Args:
        values:
            The series, one per column.

        labels:
            The label of every series.

        statistic:
            The lag-1 autocorrelation of every series in every window, the statistic of the rolling analysis.

        trend:
            Kendall's tau of the statistic of every series against time.

        variance:
            The variance of every series in every window, `nan` before the first full window.

        autocorrelation:
            The lag-1 autocorrelation of every series in every window, `nan` before the first full window.

        skewness:
            The skewness of every series in every window, `nan` before the first full window.

        reddening:
            The reddening of every series in every window, `nan` before the first full window.

        variance_trend:
            Kendall's tau of the variance of every series against time, positive when it rises.

        autocorrelation_trend:
            Kendall's tau of the autocorrelation of every series against time, positive when it rises.

        skewness_trend:
            Kendall's tau of the skewness of every series against time.

        reddening_trend:
            Kendall's tau of the reddening of every series against time.
    """

    values: Array
    labels: list[str]
    statistic: Array
    trend: Array
    variance: Array
    autocorrelation: Array
    skewness: Array
    reddening: Array
    variance_trend: Array
    autocorrelation_trend: Array
    skewness_trend: Array
    reddening_trend: Array


def early_warning(
    source: Source,
    *,
    variables: Sequence[str] | None = None,
    window: int | None = None,
    detrend: bool = True,
) -> EarlyWarningResult:
    """Return the rolling indicators of critical slowing down before a tipping point, for every series of a source.

    The main indicator is the rolling lag-one autocorrelation of detrended windows, and the others of Dakos et al.
    (2012) follow: the rolling variance and skewness, and the reddening, the ratio of the mean power of the lowest
    eighth of frequencies to that of the highest half, which rises as fluctuations slow down. The trend of every
    indicator is Kendall's tau against time.

    Args:
        source:
            The trajectory, or an array with one row per time and one column per series.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        window:
            The number of values in every rolling window.

        detrend:
            Whether to remove a linear trend in every window first.

    Returns:
        result:
            The rolling indicators of every series and their trends.

    Raises:
        AnalysisError:
            The window is missing or does not fit the series.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.physics.climate import early_warning
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(2000)
        >>> for t in range(1, 2000):
        ...     x[t] = (0.2 + 0.79 * t / 2000) * x[t - 1] + rng.standard_normal()
        >>> bool(early_warning(x, window=200).autocorrelation_trend[0] > 0.5)
        True
    """
    autocorrelation = rolling(source, kind='autocorrelation', variables=variables, window=window, detrend=detrend)
    variance = rolling(source, kind='variance', variables=variables, window=window, detrend=detrend)
    skewness = rolling(source, kind='skewness', variables=variables, window=window, detrend=detrend)
    size = int(window or 0)
    reddening = np.asarray([_reddening_of(column, size, detrend=detrend) for column in autocorrelation.values.T])
    return EarlyWarningResult(
        values=autocorrelation.values,
        labels=autocorrelation.labels,
        statistic=autocorrelation.statistic,
        trend=autocorrelation.trend,
        variance=variance.statistic,
        autocorrelation=autocorrelation.statistic,
        skewness=skewness.statistic,
        reddening=reddening,
        variance_trend=variance.trend,
        autocorrelation_trend=autocorrelation.trend,
        skewness_trend=skewness.trend,
        reddening_trend=np.asarray([_trend(row) for row in reddening]),
    )


def _reddening_of(values: Array, window: int, *, detrend: bool) -> Array:
    """Return the reddening of every window of a series, set at the last sample of the window."""
    array = np.asarray(values, dtype=float)
    found = np.full(array.size, np.nan)
    axis = np.arange(window)
    for end in range(window, array.size + 1):
        segment = array[end - window : end]
        if detrend:
            segment = segment - np.polyval(np.polyfit(axis, segment, 1), axis)
        found[end - 1] = _reddening(segment)
    return found


def _reddening(segment: Array) -> float:
    """Return the ratio of the mean power of the lowest eighth of frequencies to that of the highest half."""
    power = np.abs(np.fft.rfft(segment)[1:]) ** 2
    low = power[: max(power.size // 8, 1)]
    high = power[power.size // 2 :]
    return float(low.mean() / high.mean()) if high.mean() > 0 else float('nan')


def _trend(statistic: Array) -> float:
    """Return Kendall's tau of a statistic against time over its defined values."""
    valid = ~np.isnan(statistic)
    return float(stats.kendalltau(np.flatnonzero(valid), statistic[valid]).statistic)
