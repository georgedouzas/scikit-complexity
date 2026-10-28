"""Test the estimators moved onto libraries: power laws, displacement, and ordinal patterns."""

import itertools

import numpy as np
import pytest
from scipy import stats

from skcomplex.analysis import (
    fit_power_law,
    mean_squared_displacement,
    ordinal_patterns,
    symbolize,
)
from skcomplex.core.errors import AnalysisError


def _truncated_power_law(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return samples of `x^-1.5 exp(-x / 100)` above 1 by rejection."""
    samples = np.zeros(0)
    while samples.size < size:
        candidates = (1 - rng.random(4 * size)) ** (-1 / 0.5)
        kept = candidates[rng.random(candidates.size) < np.exp(-(candidates - 1) / 100)]
        samples = np.concatenate([samples, kept])
    return samples[:size]


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
