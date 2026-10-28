"""Test the early warning analysis of series approaching a tipping point."""

import numpy as np
import pytest

from skcomplex.analysis import EarlyWarningAnalysis
from skcomplex.core.errors import AnalysisError


def _warning(source: object, window: int, **settings: object) -> EarlyWarningAnalysis:
    """Return the early warning analysis of a source, run."""
    return EarlyWarningAnalysis(window=window, **settings).check().prepare(source).run()


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
    """Return the phosphorus of a noisy lake model whose loading rises slowly toward its fold, before it tips."""
    steps, dt = 40_000, 0.05
    loading = np.linspace(0.4, 0.65, steps)
    p = np.empty(steps)
    p[0] = 0.4
    for t in range(1, steps):
        drift = loading[t] - p[t - 1] + p[t - 1] ** 8 / (1 + p[t - 1] ** 8)
        p[t] = p[t - 1] + drift * dt + 0.005 * np.sqrt(dt) * rng.standard_normal()
    return p[::20]


def test_early_warning_same_on_two_tipping_points():
    """Test that a harvested population and a lake both show rising variance and autocorrelation before their folds."""
    for series in (_harvested_population(np.random.default_rng(6)), _lake(np.random.default_rng(7))):
        warning = _warning(series, series.size // 4)
        assert warning.variance_trend_[0] > 0.5
        assert warning.autocorrelation_trend_[0] > 0.5
        assert np.isfinite(warning.skewness_trend_[0])
        assert np.isfinite(warning.reddening_trend_[0])


def test_early_warning_rises_before_tipping():
    """Test that variance and autocorrelation rise as an AR(1) coefficient approaches 1."""
    rng = np.random.default_rng(8)
    series = np.zeros(3000)
    for t in range(1, series.size):
        series[t] = (0.1 + 0.88 * t / series.size) * series[t - 1] + rng.standard_normal()
    warning = _warning(series, 300)
    assert warning.variance_trend_[0] > 0.5
    assert warning.autocorrelation_trend_[0] > 0.5
    assert warning.variance_.shape == (1, 3000)
    assert np.isnan(warning.variance_[0, :299]).all()


def test_early_warning_rejects_windows():
    """Test that a missing window and a window longer than the series raise errors."""
    with pytest.raises(AnalysisError, match='Give a window'):
        EarlyWarningAnalysis().check()
    with pytest.raises(AnalysisError, match='must lie between'):
        _warning(np.zeros(10), 20)
