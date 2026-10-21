"""Test distribution statistics, event detection, and time-series measures."""

import warnings

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import (
    autocorrelation,
    ccdf,
    detect_events,
    early_warning,
    fit_power_law,
    gini,
    hill,
    histogram,
    hurst_exponent,
    mean_squared_displacement,
    power_spectrum,
)
from skcomplex.core.base import AnalysisError


def test_power_law_exponent_recovered():
    """Test that 10,000 samples of a power law with exponent 2.5 above 1 give 2.5 within 0.1 and p above 0.1."""
    samples = (1 - np.random.default_rng(1).random(10_000)) ** (-1 / 1.5)
    fit = fit_power_law(samples, bootstrap=50, random_state=0)
    assert fit.alpha == pytest.approx(2.5, abs=0.1)
    assert fit.p_value > 0.1
    ratio, p_value = fit.comparisons['exponential']
    assert ratio > 0
    assert p_value < 0.01


def test_exponential_sample_is_not_a_power_law():
    """Test that an exponential sample loses the comparison with an exponential tail."""
    samples = np.random.default_rng(2).exponential(5.0, 5000) + 1
    assert fit_power_law(samples, xmin=1.0, bootstrap=0).comparisons['exponential'][0] < 0


def test_fit_power_law_silences_powerlaw_warnings():
    """Test that the fits of `powerlaw` warn nothing, and that the silencing ends with the call."""
    rng = np.random.default_rng(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit_power_law(np.floor((1 - rng.random(3000)) ** (-1 / 1.5)), discrete=True, bootstrap=0)
        fit_power_law(rng.exponential(size=2000), bootstrap=0)
        assert caught == []
        warnings.warn('after the fit', UserWarning, stacklevel=1)
    assert [str(warning.message) for warning in caught] == ['after the fit']


def test_discrete_fit_of_preferential_attachment():
    """Test that the degrees of a Barabási-Albert graph have a tail exponent within 0.3 of 3."""
    degrees = np.array([degree for _, degree in nx.barabasi_albert_graph(10_000, 2, seed=0).degree()])
    assert fit_power_law(degrees, discrete=True, bootstrap=0).alpha == pytest.approx(3, abs=0.3)


def test_gini_and_hill():
    """Test the Gini coefficient of equal and concentrated values and the Hill tail index."""
    assert gini([5, 5, 5, 5]) == 0
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75)
    samples = (1 - np.random.default_rng(3).random(50_000)) ** (-1 / 1.5)
    assert hill(samples, 5000) == pytest.approx(1.5, abs=0.1)
    with pytest.raises(AnalysisError, match='k = 0'):
        hill(samples, 0)


def test_histogram_and_ccdf():
    """Test that log-binned densities integrate to 1 and the survival function starts at 1."""
    samples = (1 - np.random.default_rng(4).random(1000)) ** (-1 / 1.5)
    edges, density = histogram(samples, log=True)
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)
    _, survival = ccdf(samples)
    assert survival[0] == 1.0
    assert np.all(np.diff(survival) < 0)


def test_avalanche_sizes():
    """Test that an avalanche's size is the sum of topplings over its consecutive nonzero steps."""
    topplings = [0, 0, 3, 5, 1, 0, 2, 0, 0, 0, 7, 7]
    events = detect_events(topplings)
    assert events.sizes.tolist() == [9.0, 2.0, 14.0]
    assert events.durations.tolist() == [3.0, 1.0, 2.0]
    assert events.waiting_times.tolist() == [1.0, 3.0]
    assert events.starts.tolist() == [2.0, 6.0, 10.0]


def test_brownian_motion_diffuses_normally():
    """Test that 1000 Brownian particles have a mean squared displacement exponent within 0.05 of 1."""
    walks = np.cumsum(np.random.default_rng(5).standard_normal((2000, 1000, 2)), axis=0)
    assert mean_squared_displacement(walks).exponent == pytest.approx(1.0, abs=0.05)


def test_white_noise_has_no_memory():
    """Test that white noise has a Hurst exponent within 0.05 of 0.5 and integrated noise near 1.5."""
    noise = np.random.default_rng(6).standard_normal(50_000)
    assert hurst_exponent(noise) == pytest.approx(0.5, abs=0.05)
    assert hurst_exponent(np.cumsum(noise)) == pytest.approx(1.5, abs=0.1)


def test_autocorrelation_and_spectrum():
    """Test that an AR(1) series has autocorrelation near its coefficient and a sine has its frequency's peak."""
    rng = np.random.default_rng(7)
    series = np.zeros(20_000)
    for t in range(1, series.size):
        series[t] = 0.8 * series[t - 1] + rng.standard_normal()
    assert autocorrelation(series, max_lag=1)[1] == pytest.approx(0.8, abs=0.02)
    times = np.arange(8192) * 0.01
    frequencies, density = power_spectrum(np.sin(2 * np.pi * 3 * times), dt=0.01)
    assert frequencies[density.argmax()] == pytest.approx(3, abs=0.1)


def test_early_warning_rises_before_tipping():
    """Test that variance and autocorrelation rise as an AR(1) coefficient approaches 1."""
    rng = np.random.default_rng(8)
    series = np.zeros(3000)
    for t in range(1, series.size):
        series[t] = (0.1 + 0.88 * t / series.size) * series[t - 1] + rng.standard_normal()
    warning = early_warning(series, 300)
    assert warning.variance_trend > 0.5
    assert warning.autocorrelation_trend > 0.5
