"""Measure early warnings of tipping: rolling variance, autocorrelation, skewness, and spectral reddening."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ....analysis import RollingAnalysis

type Array = npt.NDArray[np.float64]


class EarlyWarningAnalysis(RollingAnalysis):
    """Rolling indicators of critical slowing down before a tipping point, for every series of a source.

    It is the rolling analysis of kind `'autocorrelation'` with detrended windows, whose statistic and trend are the
    lag-one autocorrelation, and it adds the other indicators of Dakos et al. (2012): the rolling variance and skewness,
    and the reddening, the ratio of the mean power of the lowest eighth of frequencies to that of the highest half,
    which rises as fluctuations slow down. The trend of every indicator is Kendall's tau against time.

    Args:
        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        window:
            The number of values in every rolling window.

        detrend:
            Whether to remove a linear trend in every window first.

    Attributes:
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
        >>> from skcomplex.models.earth.realizations import EarlyWarningAnalysis
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
        super().__init__('autocorrelation', variables, window=window, detrend=detrend)

    def _run(self) -> Self:
        """Compute the rolling indicators of every series and their trends."""
        super()._run()
        self.autocorrelation_ = self.statistic_
        self.autocorrelation_trend_ = self.trend_
        self.variance_ = self._indicator(self._statistic, 'variance')
        self.variance_trend_ = np.asarray([self._trend(row) for row in self.variance_])
        self.skewness_ = self._indicator(self._statistic, 'skewness')
        self.skewness_trend_ = np.asarray([self._trend(row) for row in self.skewness_])
        self.reddening_ = self._indicator(self._reddening, 'reddening')
        self.reddening_trend_ = np.asarray([self._trend(row) for row in self.reddening_])
        return self

    def _indicator(self, statistic: Callable[[Array, str], float], kind: str) -> Array:
        """Return a rolling statistic of every series, one row per series."""
        window = int(self.window or 0)
        return np.asarray(
            [self._rolling(column, window, statistic, kind, detrend=self.detrend) for column in self.values_.T],
        )

    @staticmethod
    def _reddening(segment: Array, kind: str) -> float:
        """Return the ratio of the mean power of the lowest eighth of frequencies to that of the highest half."""
        power = np.abs(np.fft.rfft(segment)[1:]) ** 2
        low = power[: max(power.size // 8, 1)]
        high = power[power.size // 2 :]
        return float(low.mean() / high.mean()) if high.mean() > 0 else float('nan')
