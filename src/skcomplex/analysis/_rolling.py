"""Measure statistics of series in rolling windows: the variance, the lag-one autocorrelation, and the skewness."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
KINDS = ('variance', 'autocorrelation', 'skewness')
MIN_WINDOW = 3


class RollingAnalysis(BaseAnalysis):
    """Statistic of every series of a source in rolling windows, of one kind, and its trend.

    Every window holds the `window` values that end at a sample, possibly with a linear trend removed first, and the
    statistic of the window is set at that sample. The trend of the statistic is Kendall's tau against time.

    - `'variance'`: the variance of the window.
    - `'autocorrelation'`: the lag-one autocorrelation of the window.
    - `'skewness'`: the skewness of the window.

    Args:
        kind:
            The statistic to compute, one of the kinds above.

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

        statistic_:
            The statistic of every series in every window, one row per series, `nan` before the first full window.

        trend_:
            Kendall's tau of the statistic of every series against time, positive when it rises.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import RollingAnalysis
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(2000)
        >>> for t in range(1, 2000):
        ...     x[t] = (0.2 + 0.79 * t / 2000) * x[t - 1] + rng.standard_normal()
        >>> rolling = RollingAnalysis('autocorrelation', window=200, detrend=True).check().prepare(x).run()
        >>> bool(rolling.trend_[0] > 0.5)
        True
    """

    def __init__(
        self,
        kind: str = 'variance',
        variables: Sequence[str] | None = None,
        *,
        window: int | None = None,
        detrend: bool = False,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.window = window
        self.detrend = detrend

    def _check(self) -> Self:
        """Validate the kind and the window."""
        self._check_choice('kind', self.kind, KINDS)
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
        """Compute the rolling statistic of every series and its trend."""
        window = int(self.window or 0)
        columns = self.values_.T
        self.statistic_ = np.asarray(
            [self._rolling(column, window, self._statistic, self.kind, detrend=self.detrend) for column in columns],
        )
        self.trend_ = np.asarray([self._trend(row) for row in self.statistic_])
        return self

    @staticmethod
    def _statistic(segment: Array, kind: str) -> float:
        """Return the statistic of one kind of a window."""
        if kind == 'variance':
            return float(segment.var())
        if kind == 'autocorrelation':
            return float(np.corrcoef(segment[:-1], segment[1:])[0, 1])
        return float(stats.skew(segment))

    @staticmethod
    def _trend(statistic: Array) -> float:
        """Return Kendall's tau of a statistic against time over its defined values."""
        valid = ~np.isnan(statistic)
        return float(stats.kendalltau(np.flatnonzero(valid), statistic[valid]).statistic)

    @staticmethod
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
        array = BaseAnalysis._read_real(values, 'values')
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
