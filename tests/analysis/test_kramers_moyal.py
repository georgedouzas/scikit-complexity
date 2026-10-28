"""Test the Kramers-Moyal analysis of drift and diffusion."""

import numpy as np
import pytest

from skcomplex.analysis import KramersMoyalAnalysis
from skcomplex.core.errors import AnalysisError


def test_kramers_moyal_ornstein_uhlenbeck_coefficients():
    """Test that 10^6 steps of an Ornstein-Uhlenbeck process give drift -x within 0.1 and diffusion 1 within 0.05."""
    rng = np.random.default_rng(2)
    dt = 0.02
    noise = rng.standard_normal(1_000_000) * np.sqrt(2 * dt)
    values = np.empty(noise.size)
    values[0] = 0.0
    for t in range(1, values.size):
        values[t] = values[t - 1] - values[t - 1] * dt + noise[t]
    coefficients = KramersMoyalAnalysis(dt=dt, bins=41).check().prepare(values).run()
    inside = np.abs(coefficients.positions_) < 1.5
    assert np.max(np.abs(coefficients.drift_[inside] + coefficients.positions_[inside])) < 0.1
    assert np.max(np.abs(coefficients.diffusion_[inside] - 1)) < 0.05


def test_kramers_moyal_reads_one_series():
    """Test that two series and a negative step raise errors."""
    with pytest.raises(AnalysisError, match='reads one series'):
        KramersMoyalAnalysis().check().prepare(np.zeros((10, 2)))
    with pytest.raises(AnalysisError, match='dt'):
        KramersMoyalAnalysis(dt=-1.0).check()
