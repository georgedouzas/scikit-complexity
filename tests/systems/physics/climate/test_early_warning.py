"""Test the `_early_warning` module of `skcomplex.systems.physics.climate`."""

import numpy as np

from skcomplex.core.analysis import rolling
from skcomplex.systems.physics.climate import early_warning


def _approaching_fold(rng: np.random.Generator) -> np.ndarray:
    """Return an AR(1) series whose coefficient rises towards 1, slowing down as before a fold."""
    series = np.zeros(3000)
    for t in range(1, series.size):
        series[t] = (0.1 + 0.88 * t / series.size) * series[t - 1] + rng.standard_normal()
    return series


def test_early_warning_equals_the_rolling_analysis():
    """Test that every indicator equals the detrended rolling statistic of its kind, the main one autocorrelation."""
    series = _approaching_fold(np.random.default_rng(8))
    warning = early_warning(series, window=300)
    for kind in ('variance', 'autocorrelation', 'skewness'):
        found = rolling(series, kind=kind, window=300, detrend=True)
        np.testing.assert_array_equal(getattr(warning, kind), found.statistic)
        np.testing.assert_array_equal(getattr(warning, f'{kind}_trend'), found.trend)
    np.testing.assert_array_equal(warning.statistic, warning.autocorrelation)


def test_early_warning_reddens_before_tipping():
    """Test that variance, autocorrelation, and reddening rise as the series slows down."""
    warning = early_warning(_approaching_fold(np.random.default_rng(8)), window=300)
    assert warning.variance_trend[0] > 0.5
    assert warning.autocorrelation_trend[0] > 0.5
    assert warning.reddening_trend[0] > 0
    assert warning.reddening.shape == (1, 3000)
    assert np.isnan(warning.reddening[0, :299]).all()
