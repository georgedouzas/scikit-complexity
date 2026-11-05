"""Test the `_early_warning` module of `skcomplex.systems.physics.climate`."""

import numpy as np

from skcomplex.core.analysis import RollingAnalysis
from skcomplex.systems.physics.climate import EarlyWarningAnalysis


def _approaching_fold(rng: np.random.Generator) -> np.ndarray:
    """Return an AR(1) series whose coefficient rises towards 1, slowing down as before a fold."""
    series = np.zeros(3000)
    for t in range(1, series.size):
        series[t] = (0.1 + 0.88 * t / series.size) * series[t - 1] + rng.standard_normal()
    return series


def test_early_warning_equals_the_rolling_analysis():
    """Test that every indicator equals the detrended rolling analysis of its kind, the main one autocorrelation."""
    series = _approaching_fold(np.random.default_rng(8))
    warning = EarlyWarningAnalysis(window=300).prepare(series).run()
    for kind in ('variance', 'autocorrelation', 'skewness'):
        rolling = RollingAnalysis(kind, window=300, detrend=True).prepare(series).run()
        np.testing.assert_array_equal(getattr(warning, f'{kind}_'), rolling.statistic_)
        np.testing.assert_array_equal(getattr(warning, f'{kind}_trend_'), rolling.trend_)
    np.testing.assert_array_equal(warning.statistic_, warning.autocorrelation_)
    assert warning.kind == 'autocorrelation'


def test_early_warning_reddens_before_tipping():
    """Test that variance, autocorrelation, and reddening rise as the series slows down."""
    warning = EarlyWarningAnalysis(window=300).prepare(_approaching_fold(np.random.default_rng(8))).run()
    assert warning.variance_trend_[0] > 0.5
    assert warning.autocorrelation_trend_[0] > 0.5
    assert warning.reddening_trend_[0] > 0
    assert warning.reddening_.shape == (1, 3000)
    assert np.isnan(warning.reddening_[0, :299]).all()
