"""Test the estimators moved onto libraries: power laws, fluctuations, displacement, warnings, and ordinal patterns."""

import itertools
import time

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.analysis import (
    autocorrelation_time,
    detrended_fluctuation,
    early_warning,
    fit_power_law,
    hurst_exponent,
    lempel_ziv_complexity,
    mean_squared_displacement,
    ordinal_patterns,
    permutation_entropy,
    symbolize,
)
from skcomplex.core.base import AnalysisError


def _truncated_power_law(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return samples of `x^-1.5 exp(-x / 100)` above 1 by rejection."""
    samples = np.zeros(0)
    while samples.size < size:
        candidates = (1 - rng.random(4 * size)) ** (-1 / 0.5)
        kept = candidates[rng.random(candidates.size) < np.exp(-(candidates - 1) / 100)]
        samples = np.concatenate([samples, kept])
    return samples[:size]


def _fractional_gaussian_noise(size: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Return fractional Gaussian noise by the circulant embedding of Davies and Harte."""
    lags = np.arange(size + 1, dtype=float)
    covariance = 0.5 * (np.abs(lags + 1) ** (2 * hurst) - 2 * lags ** (2 * hurst) + np.abs(lags - 1) ** (2 * hurst))
    circulant = np.concatenate([covariance, covariance[-2:0:-1]])
    eigenvalues = np.clip(np.fft.fft(circulant).real, 0, None)
    weights = np.sqrt(eigenvalues / circulant.size) * (
        rng.standard_normal(circulant.size) + 1j * rng.standard_normal(circulant.size)
    )
    return np.fft.fft(weights).real[:size]


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


def test_fit_power_law_discrete_samples():
    """Test that 10,000 discrete power-law samples above 1 give their exponent with the exact discrete likelihood."""
    samples = stats.zipf(2.5).rvs(10_000, random_state=np.random.default_rng(0))
    fit = fit_power_law(samples, discrete=True, bootstrap=0)
    assert fit.alpha == pytest.approx(2.5, abs=0.1)


def test_fit_power_law_truncated_tail():
    """Test that a power law with an exponential cutoff prefers the truncated power law with p below 0.05."""
    samples = _truncated_power_law(10_000, np.random.default_rng(1))
    ratio, p_value = fit_power_law(samples, xmin=1.0, bootstrap=0).comparisons['truncated_power_law']
    assert ratio < 0
    assert p_value < 0.05


def test_fit_power_law_upper_cutoff():
    """Test that an upper cutoff limits the tail and is reported."""
    samples = (1 - np.random.default_rng(2).random(5000)) ** (-1 / 1.5)
    fit = fit_power_law(samples, xmin=1.0, xmax=100.0, bootstrap=0)
    assert fit.xmax == 100.0
    assert fit.tail == int(np.sum(samples <= 100))


def test_fit_power_law_matches_previous_estimates():
    """Test that the fixed-cutoff fit equals the previous in-house estimate on a fixed seed."""
    samples = (1 - np.random.default_rng(7).random(5000)) ** (-1 / 1.5)
    fit = fit_power_law(samples, xmin=1.0, bootstrap=0)
    assert fit.alpha == pytest.approx(2.5147885507011036, abs=1e-8)
    assert fit.comparisons["exponential"][0] == pytest.approx(1619.996186992908, rel=1e-6)


def test_mean_squared_displacement_brownian_motion():
    """Test that 1000 Brownian particles give an exponent within 0.05 of 1 and their diffusion coefficient."""
    steps = np.random.default_rng(3).normal(0, np.sqrt(2 * 0.5 * 0.1), (2000, 1000, 2))
    displacement = mean_squared_displacement(np.cumsum(steps, axis=0), dt=0.1)
    assert displacement.exponent == pytest.approx(1, abs=0.05)
    assert displacement.diffusion == pytest.approx(0.5, rel=0.05)
    assert displacement.lags.size == 1999


def test_mean_squared_displacement_rejects_an_empty_range():
    """Test that a fit range without lags raises an error."""
    with pytest.raises(AnalysisError, match='fit range'):
        mean_squared_displacement(np.zeros((10, 2)), fit_range=(100.0, 200.0))


def test_hurst_exponent_white_noise_memory():
    """Test that white noise has a Hurst exponent within 0.05 of 0.5 by both estimators."""
    noise = np.random.default_rng(4).standard_normal(20_000)
    assert hurst_exponent(noise) == pytest.approx(0.5, abs=0.05)
    assert hurst_exponent(noise, method='rs') == pytest.approx(0.5, abs=0.05)


def test_detrended_fluctuation_persistent_and_integrated_noise():
    """Test that fractional Gaussian noise with H = 0.7 gives 0.7 and a random walk 1.5, within 0.05."""
    persistent = _fractional_gaussian_noise(20_000, 0.7, np.random.default_rng(11))
    walk = np.cumsum(np.random.default_rng(5).standard_normal(20_000))
    assert detrended_fluctuation(persistent).exponent == pytest.approx(0.7, abs=0.05)
    assert detrended_fluctuation(walk).exponent == pytest.approx(1.5, abs=0.05)


def test_detrended_fluctuation_rejects_short_series_and_unknown_methods():
    """Test that short series and unknown methods raise errors."""
    with pytest.raises(AnalysisError, match='at least 100'):
        detrended_fluctuation(np.zeros(10))
    with pytest.raises(AnalysisError, match='Unknown method'):
        hurst_exponent(np.random.default_rng(0).standard_normal(500), method='wavelet')


def test_early_warning_same_on_two_tipping_points():
    """Test that a harvested population and a lake both show rising variance and autocorrelation before their folds."""
    for series in (_harvested_population(np.random.default_rng(6)), _lake(np.random.default_rng(7))):
        warning = early_warning(series, series.size // 4)
        assert warning.variance_trend > 0.5
        assert warning.autocorrelation_trend > 0.5
        assert np.isfinite(warning.skewness_trend)
        assert np.isfinite(warning.reddening_trend)


def test_autocorrelation_time_of_an_autoregression():
    """Test that an autoregression with coefficient 0.9 has an autocorrelation time near 9.5."""
    rng = np.random.default_rng(8)
    x = np.zeros(100_000)
    for t in range(1, x.size):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    assert autocorrelation_time(x) == pytest.approx(9.5, rel=0.15)


def test_symbolize_long_ordinal_patterns():
    """Test that ordinal patterns of length 12 of white noise give one symbol per distinct rank pattern."""
    noise = np.random.default_rng(9).standard_normal(20_000)
    symbols = symbolize(noise, 'ordinal', order=12)
    patterns = [
        tuple(np.argsort(window, kind='stable')) for window in np.lib.stride_tricks.sliding_window_view(noise, 12)
    ]
    pairs = set(zip(symbols.tolist(), patterns, strict=True))
    assert len(pairs) == len(set(symbols.tolist())) == len(set(patterns))


def test_ordinal_patterns_rank_every_permutation():
    """Test that the patterns of length 3 are the lexicographic ranks 0 to 5."""
    ranks = []
    for permutation in itertools.permutations(range(3)):
        values = np.empty(3)
        values[list(permutation)] = np.arange(3)
        ranks.append(int(ordinal_patterns(values, 3)[0]))
    assert ranks == list(range(6))


def _previous_draws() -> tuple[list[np.ndarray], np.ndarray]:
    """Return the fixed sequences and the noise the previous in-house estimators were evaluated on."""
    rng = np.random.default_rng(7)
    sequences = [rng.integers(0, 2, 500), rng.integers(0, 4, 300), rng.integers(0, 3, 1000)]
    return sequences, rng.standard_normal(3000)


def test_permutation_entropy_matches_previous_values_at_delay_1():
    """Test that the permutation entropy at delay 1 equals the previous in-house estimate on a fixed seed."""
    _, values = _previous_draws()
    assert permutation_entropy(values, 4, normalize=False) == pytest.approx(4.5814617588460695, abs=1e-8)
    assert permutation_entropy(values, 4, delay=2) == pytest.approx(1, abs=0.01)


def test_permutation_entropy_rejects_unknown_entropies():
    """Test that an unknown entropy raises an error."""
    with pytest.raises(AnalysisError, match='Unknown entropy'):
        permutation_entropy(np.arange(10.0), entropy='wiener')


def test_lempel_ziv_complexity_matches_previous_values():
    """Test that the phrase counts equal the previous in-house parsing on fixed sequences."""
    sequences, _ = _previous_draws()
    assert [lempel_ziv_complexity(sequence, normalize=False) for sequence in sequences] == [62.0, 75.0, 165.0]


def test_lempel_ziv_complexity_of_a_long_sequence():
    """Test that 10^5 fair coin flips have a normalized complexity within 0.1 of 1."""
    flips = np.random.default_rng(10).integers(0, 2, 100_000)
    assert lempel_ziv_complexity(flips) == pytest.approx(1, abs=0.1)


@pytest.mark.benchmark
def test_lempel_ziv_complexity_of_a_long_sequence_is_fast():
    """Test that the complexity of 10^5 coin flips returns within one second on an idle machine."""
    flips = np.random.default_rng(10).integers(0, 2, 100_000)
    lempel_ziv_complexity(flips[:100])
    start = time.perf_counter()
    lempel_ziv_complexity(flips)
    assert time.perf_counter() - start < 1
