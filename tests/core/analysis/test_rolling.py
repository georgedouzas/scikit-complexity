"""Test the rolling variance, autocorrelation, and skewness of series, such as series approaching a tipping point."""

import numpy as np
import pytest

from skcomplex.core.analysis import RollingResult, rolling
from skcomplex.core.errors import AnalysisError


def _rolling(kind: str, source: object, window: int, **settings: object) -> RollingResult:
    """Return the rolling statistic of a source for one kind."""
    return rolling(source, kind=kind, window=window, **settings)


def _harvested_population(rng: np.random.Generator) -> np.ndarray:
    """Return a noisy logistic population with harvesting that rises slowly toward its fold, before it tips."""
    steps, dt = 40_000, 0.05
    harvest = np.linspace(1.0, 2.55, steps)
    x = np.empty(steps)
    x[0] = 7.6
    for t in range(1, steps):
        drift = x[t - 1] * (1 - x[t - 1] / 10) - harvest[t] * x[t - 1] ** 2 / (1 + x[t - 1] ** 2)
        x[t] = x[t - 1] + drift * dt + 0.05 * np.sqrt(dt) * rng.standard_normal()
    return x[::20]


def _lake(rng: np.random.Generator) -> np.ndarray:
    """Return the phosphorus of a noisy lake system whose loading rises slowly toward its fold, before it tips."""
    steps, dt = 40_000, 0.05
    loading = np.linspace(0.4, 0.65, steps)
    p = np.empty(steps)
    p[0] = 0.4
    for t in range(1, steps):
        drift = loading[t] - p[t - 1] + p[t - 1] ** 8 / (1 + p[t - 1] ** 8)
        p[t] = p[t - 1] + drift * dt + 0.005 * np.sqrt(dt) * rng.standard_normal()
    return p[::20]


def test_rolling_same_on_two_tipping_points():
    """Test that a harvested population and a lake both show rising variance and autocorrelation before their folds."""
    for series in (_harvested_population(np.random.default_rng(6)), _lake(np.random.default_rng(7))):
        window = series.size // 4
        assert _rolling('variance', series, window, detrend=True).trend[0] > 0.5
        assert _rolling('autocorrelation', series, window, detrend=True).trend[0] > 0.5
        assert np.isfinite(_rolling('skewness', series, window, detrend=True).trend[0])


def test_rolling_rises_before_tipping():
    """Test that variance and autocorrelation rise as an AR(1) coefficient approaches 1."""
    rng = np.random.default_rng(8)
    series = np.zeros(3000)
    for t in range(1, series.size):
        series[t] = (0.1 + 0.88 * t / series.size) * series[t - 1] + rng.standard_normal()
    variance = _rolling('variance', series, 300)
    assert variance.trend[0] > 0.5
    assert _rolling('autocorrelation', series, 300).trend[0] > 0.5
    assert variance.statistic.shape == (1, 3000)
    assert np.isnan(variance.statistic[0, :299]).all()
    assert variance.statistic[0, 299] == pytest.approx(series[:300].var())


def test_rolling_rejects_kinds_and_windows():
    """Test that an unknown kind, a missing window, and a window longer than the series raise errors."""
    with pytest.raises(AnalysisError, match='variance, autocorrelation, skewness'):
        rolling(np.zeros(20), kind='kurtosis', window=10)
    with pytest.raises(AnalysisError, match='Give a window'):
        rolling(np.zeros(20))
    with pytest.raises(AnalysisError, match='must lie between'):
        _rolling('variance', np.zeros(10), 20)
