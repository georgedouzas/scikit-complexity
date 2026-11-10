"""Test the `_outbreaks` module of `skcomplex.systems.biology.epidemiology`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import HeterogeneousMeanFieldSIRSystem


def _degree_moments(exponent, k_min, k_max):
    """Return the degrees, their probabilities, and the first two moments of a power-law degree distribution."""
    degrees = np.arange(k_min, k_max + 1, dtype=float)
    probabilities = degrees**-exponent / np.sum(degrees**-exponent)
    return degrees, probabilities, np.sum(degrees * probabilities), np.sum(degrees**2 * probabilities)


@pytest.mark.parametrize(('factor', 'below'), [(0.5, True), (2.0, False)])
def test_heterogeneous_mean_field_sir_epidemic_threshold(factor, below):
    """Test that the final epidemic size is below 1% below the threshold <k>/<k^2> and above 5% above it."""
    _, _, first, second = _degree_moments(2.5, 3, 100)
    beta = factor * first / second
    trajectory = simulate(
        HeterogeneousMeanFieldSIRSystem(beta=beta, mu=1.0),
        dt=1.0,
        until=200,
    )
    final_size = trajectory.get_values('recovered')[-1]
    assert trajectory.get_values('infected')[-1] < 1e-6
    assert (final_size < 0.01) if below else (final_size > 0.05)


def test_heterogeneous_mean_field_sir_early_growth_rate():
    """Test that theta grows at the rate (beta (<k^2> - <k>) - mu <k>) / <k> of the linearized equation (6.30)."""
    degrees, probabilities, first, second = _degree_moments(2.5, 3, 100)
    beta, mu = 0.13, 1.0
    system = HeterogeneousMeanFieldSIRSystem(beta=beta, mu=mu, infected=1e-9)
    trajectory = simulate(
        system,
        dt=0.5,
        until=12,
    )
    theta = trajectory.get_values('network.i') @ ((degrees - 1) * probabilities) / first
    selected = trajectory.times >= 4
    rate = np.polyfit(trajectory.times[selected], np.log(theta[selected]), 1)[0]
    assert rate == pytest.approx((beta * (second - first) - mu * first) / first, rel=0.03)
