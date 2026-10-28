"""Test the estimators moved onto libraries: power laws."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.analysis import fit_power_law


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
