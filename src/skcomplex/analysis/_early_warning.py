"""Measure early warnings of tipping: rolling variance, autocorrelation, skewness, and spectral reddening."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
MIN_WINDOW = 3


class EarlyWarningAnalysis(BaseAnalysis):
    """Rolling indicators of critical slowing down before a tipping point, for every series of a source.

    In every rolling window, possibly detrended, the analysis measures the variance, the lag-1 autocorrelation, the
    skewness, and the reddening, the ratio of the mean power of the lowest eighth of frequencies to that of the highest
    half, which rises as fluctuations slow down. The trend of every indicator is Kendall's tau against time.

    Args:
        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        window:
            The number of values in every rolling window.

        detrend:
            Whether to remove a linear trend in every window first.

    Attributes:
        values_:
            The series, one per column.

        labels_:
            The label of every series.

        variance_:
            The variance of every series in every window, `nan` before the first full window.

        autocorrelation_:
            The lag-1 autocorrelation of every series in every window, `nan` before the first full window.

        skewness_:
            The skewness of every series in every window, `nan` before the first full window.

        reddening_:
            The reddening of every series in every window, `nan` before the first full window.

        variance_trend_:
            Kendall's tau of the variance of every series against time, positive when it rises.

        autocorrelation_trend_:
            Kendall's tau of the autocorrelation of every series against time, positive when it rises.

        skewness_trend_:
            Kendall's tau of the skewness of every series against time.

        reddening_trend_:
            Kendall's tau of the reddening of every series against time.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import EarlyWarningAnalysis
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(2000)
        >>> for t in range(1, 2000):
        ...     x[t] = (0.2 + 0.79 * t / 2000) * x[t - 1] + rng.standard_normal()
        >>> bool(EarlyWarningAnalysis(window=200).check().prepare(x).run().autocorrelation_trend_[0] > 0.5)
        True
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        window: int | None = None,
        detrend: bool = True,
    ) -> None:
        self.variables = variables
        self.window = window
        self.detrend = detrend

    def _check(self) -> Self:
        """Validate the window."""
        if self.window is None or self.window < MIN_WINDOW:
            error_msg = f'{type(self).__name__} has window={self.window!r}. Give a window of {MIN_WINDOW} or more.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the series as columns."""
        series = self._read_series(source, self.variables)
        self.values_ = series.values
        self.labels_ = series.labels
        return self

    def _run(self) -> Self:
        """Compute the rolling indicators of every series and their trends."""
        window = int(self.window or 0)
        indicators = np.asarray([self._indicators(column, window, detrend=self.detrend) for column in self.values_.T])
        self.variance_ = indicators[:, 0]
        self.autocorrelation_ = indicators[:, 1]
        self.skewness_ = indicators[:, 2]
        self.reddening_ = indicators[:, 3]
        self.variance_trend_ = np.asarray([self._trend(row) for row in self.variance_])
        self.autocorrelation_trend_ = np.asarray([self._trend(row) for row in self.autocorrelation_])
        self.skewness_trend_ = np.asarray([self._trend(row) for row in self.skewness_])
        self.reddening_trend_ = np.asarray([self._trend(row) for row in self.reddening_])
        return self

    @staticmethod
    def _reddening(segment: Array) -> float:
        """Return the ratio of the mean power of the lowest eighth of frequencies to that of the highest half."""
        power = np.abs(np.fft.rfft(segment)[1:]) ** 2
        low = power[: max(power.size // 8, 1)]
        high = power[power.size // 2 :]
        return float(low.mean() / high.mean()) if high.mean() > 0 else float('nan')

    @staticmethod
    def _trend(indicator: Array) -> float:
        """Return Kendall's tau of an indicator against time over its defined values."""
        valid = ~np.isnan(indicator)
        return float(stats.kendalltau(np.flatnonzero(valid), indicator[valid]).statistic)

    @staticmethod
    def _indicators(values: npt.ArrayLike, window: int, *, detrend: bool) -> Array:
        """Return the rolling variance, lag-1 autocorrelation, skewness, and reddening of a series, one row each."""
        array = BaseAnalysis._read_real(values, 'values')
        if not MIN_WINDOW <= window <= array.size:
            error_msg = (
                f'window = {window} must lie between {MIN_WINDOW} and the length of the series, {array.size}. '
                'Use a window within that range.'
            )
            raise AnalysisError(error_msg)
        indicators = np.full((4, array.size), np.nan)
        axis = np.arange(window)
        for end in range(window, array.size + 1):
            segment = array[end - window : end]
            if detrend:
                segment = segment - np.polyval(np.polyfit(axis, segment, 1), axis)
            indicators[:, end - 1] = (
                segment.var(),
                np.corrcoef(segment[:-1], segment[1:])[0, 1],
                stats.skew(segment),
                EarlyWarningAnalysis._reddening(segment),
            )
        return indicators
