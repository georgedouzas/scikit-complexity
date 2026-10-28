"""Test event detection and time-series measures."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.analysis import (
    FitAnalysis,
    autocorrelation,
    degrees,
    detect_events,
    early_warning,
    hurst_exponent,
    mean_squared_displacement,
    power_spectrum,
)
from skcomplex.core.spaces import NetworkSpace


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


@pytest.mark.slow
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


def test_barabasi_albert_degree_exponent():
    """Test that the degree distribution of a Barabási-Albert graph has an exponent within 0.3 of 3."""
    graph = nx.barabasi_albert_graph(10_000, 2, seed=0)
    fit = FitAnalysis(discrete=True, bootstrap=0).check().prepare(degrees(NetworkSpace(graph))).run()
    assert fit.exponent_ == pytest.approx(3, abs=0.3)
