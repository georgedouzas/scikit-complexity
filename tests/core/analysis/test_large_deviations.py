"""Test generating functions, rate functions, and exponential reweighting."""

import math

import numpy as np
import pytest
from scipy import special

from skcomplex.core.analysis import (
    block_generating_function,
    histogram_rate_function,
    legendre_transform,
    reweight,
    tilted_generating_function,
)
from skcomplex.core.base import AnalysisError


def _mean_field_magnetizations(size: int, beta: float, field: float, samples: int, seed: int) -> np.ndarray:
    """Return exact samples of the magnetization per spin of the mean-field Ising model."""
    totals = np.arange(-size, size + 1, 2)
    logs = (
        special.gammaln(size + 1) - special.gammaln((size + totals) / 2 + 1) - special.gammaln((size - totals) / 2 + 1)
    )
    logs = logs + beta * totals**2 / (2 * size) + beta * field * totals
    weights = np.exp(logs - logs.max())
    return np.random.default_rng(seed).choice(totals, samples, p=weights / weights.sum()) / size


def test_tilted_and_block_generating_functions_of_poisson_counting():
    """Test that both generating functions of a Poisson count with rate 2 equal 2 (e^k - 1) and the rate at 3."""
    tilts = np.linspace(-1, 1, 9)
    exact = 2 * (np.exp(tilts) - 1)
    np.testing.assert_allclose(tilted_generating_function([[2.0]], [[1.0]], tilts), exact, rtol=0.02, atol=1e-12)
    counts = np.random.default_rng(0).poisson(0.2, 10_000_000)
    np.testing.assert_allclose(block_generating_function(counts, tilts, 10, 0.1), exact, rtol=0.02, atol=0.005)
    grid = np.linspace(-3, 3, 6001)
    rate = legendre_transform(grid, tilted_generating_function([[2.0]], [[1.0]], grid), 3.0)[0]
    assert rate == pytest.approx(3 * math.log(3 / 2) - 1, rel=0.02)


def test_tilted_generating_function_gallavotti_cohen_symmetry():
    """Test that the current of the driven ring satisfies lambda(k) = lambda(-k - ln 2)."""
    ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
    current = np.roll(np.eye(5), 1, axis=1) - np.roll(np.eye(5), -1, axis=1)
    tilts = np.linspace(-2, 1, 13)
    np.testing.assert_allclose(
        tilted_generating_function(ring, current, tilts),
        tilted_generating_function(ring, current, -tilts - math.log(2)),
        atol=1e-8,
    )


def test_histogram_rate_function_mean_field_order_parameter():
    """Test that histograms at sizes 50, 100, and 200 give the exact rate function above the critical temperature."""
    beta = 0.5
    samples = {size: _mean_field_magnetizations(size, beta, 0.0, 10_000_000, size) for size in (50, 100, 200)}
    edges = np.arange(-1.02, 1.03, 0.04)
    rate = histogram_rate_function(samples, edges)
    m = rate.values
    exact = (1 + m) / 2 * np.log1p(m) + (1 - m) / 2 * np.log1p(-m) - beta * m**2 / 2
    inside = (np.abs(m) < 0.5) & np.isfinite(rate.rates)
    assert np.max(np.abs(rate.rates[inside] - (exact[inside] - exact[inside].min()))) < 0.02
    with pytest.raises(AnalysisError, match='at least 2 sizes'):
        histogram_rate_function({50: samples[50]})


def test_reweight_magnetization_to_a_field():
    """Test that the magnetization at field 0 reweighted to field 0.05 matches a run at field 0.05."""
    size, beta, field = 100, 0.8, 0.05
    zero = _mean_field_magnetizations(size, beta, 0.0, 2_000_000, 1)
    target = _mean_field_magnetizations(size, beta, field, 2_000_000, 2)
    reweighted = reweight(zero, beta * field * size * zero)
    assert reweighted.mean == pytest.approx(target.mean(), abs=0.02)
    assert reweighted.effective_size > 1000


def test_block_generating_function_needs_two_blocks():
    """Test that a run shorter than two blocks raises an error."""
    with pytest.raises(AnalysisError, match='fewer than 2 blocks'):
        block_generating_function([1.0, 2.0], [0.5], 5)
