"""Test the `_outbreaks` module of `skcomplex.models.biology.epidemiology`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.biology.epidemiology import heterogeneous_mean_field_sir


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
    trajectory = (
        Simulation(
            built := heterogeneous_mean_field_sir(beta=beta, mu=1.0),
            dt=1.0,
            observables=describe('heterogeneous_mean_field_sir').observables_of(built),
        )
        .run(until=200)
        .trajectory_
    )
    final_size = trajectory['recovered'][-1]
    assert trajectory['infected'][-1] < 1e-6
    assert (final_size < 0.01) if below else (final_size > 0.05)


def test_heterogeneous_mean_field_sir_early_growth_rate():
    """Test that theta grows at the rate (beta (<k^2> - <k>) - mu <k>) / <k> of the linearized equation (6.30)."""
    degrees, probabilities, first, second = _degree_moments(2.5, 3, 100)
    beta, mu = 0.13, 1.0
    system = heterogeneous_mean_field_sir(beta=beta, mu=mu, infected=1e-9)
    trajectory = (
        Simulation(system, dt=0.5, observables=describe('heterogeneous_mean_field_sir').observables_of(system))
        .run(until=12)
        .trajectory_
    )
    theta = trajectory['network.i'] @ ((degrees - 1) * probabilities) / first
    selected = trajectory.times >= 4
    rate = np.polyfit(trajectory.times[selected], np.log(theta[selected]), 1)[0]
    assert rate == pytest.approx((beta * (second - first) - mu * first) / first, rel=0.03)
