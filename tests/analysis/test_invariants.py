"""Test Lyapunov-derived invariants and invariant densities."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.analysis import (
    invariant_density,
    kaplan_yorke_dimension,
    kolmogorov_sinai_entropy,
    largest_lyapunov_exponent,
    lyapunov_spectrum,
    ulam_chain,
)
from skcomplex.models.mathematics.dynamical_systems import henon_map, logistic_map, lorenz


def _lorenz_states(rho: float, step: float, duration: float) -> np.ndarray:
    """Return states of the Lorenz-63 flow every `step` after a transient of 100, one row per sample."""

    def drift(_, state):
        x, y, z = state
        return [10 * (y - x), x * (rho - z) - y, x * y - 8 / 3 * z]

    times = np.arange(100, 100 + duration, step)
    solution = solve_ivp(drift, (0, times[-1]), [1, 1, 1], t_eval=times, rtol=1e-9, atol=1e-9)
    return solution.y.T


def _henon(size: int) -> np.ndarray:
    """Return iterates of the Hénon map after a transient of 100, one row per iterate."""
    points = np.empty((size + 100, 2))
    x, y = 0.1, 0.0
    for index in range(points.shape[0]):
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        points[index] = x, y
    return points[100:]


def _logistic(r: float, size: int) -> np.ndarray:
    """Return iterates of the logistic map after a transient of 500."""
    values = np.empty(size + 500)
    values[0] = 0.3
    for t in range(1, values.size):
        values[t] = r * values[t - 1] * (1 - values[t - 1])
    return values[500:]


def test_kolmogorov_sinai_entropy_of_a_map_and_a_flow():
    """Test that the logistic map at r = 4 gives ln 2 and the Lorenz flow 0.906."""
    logistic = lyapunov_spectrum(logistic_map(r=4.0, x=0.3), until=20_000, dt=1)
    flow = lyapunov_spectrum(lorenz(), until=500)
    assert kolmogorov_sinai_entropy(logistic) == pytest.approx(math.log(2), abs=0.01)
    assert kolmogorov_sinai_entropy(flow) == pytest.approx(0.906, abs=0.05)


def test_kaplan_yorke_dimension_of_a_map_and_a_flow():
    """Test that the Hénon map gives 1.26 and the Lorenz flow 2.06."""
    henon = lyapunov_spectrum(henon_map(), until=20_000, dt=1)
    flow = lyapunov_spectrum(lorenz(), until=500)
    assert kaplan_yorke_dimension(henon) == pytest.approx(1.26, abs=0.02)
    assert kaplan_yorke_dimension(flow) == pytest.approx(2.06, abs=0.03)
    assert kaplan_yorke_dimension([-1.0, -2.0]) == 0
    assert kaplan_yorke_dimension([0.5, 0.0]) == 2


@pytest.mark.slow
def test_largest_lyapunov_exponent_from_data():
    """Test the largest exponents of the logistic map at r = 4 and of the Lorenz x from recorded series."""
    logistic = largest_lyapunov_exponent(_logistic(4.0, 10_000), dimension=2, trajectory_length=4, min_separation=10)
    assert logistic == pytest.approx(math.log(2), abs=0.1)
    series = _lorenz_states(28.0, 0.05, 300.0)[:6000, 0]
    flow = largest_lyapunov_exponent(
        series,
        0.05,
        dimension=5,
        delay=2,
        trajectory_length=60,
        min_separation=30,
        skip=10,
    )
    assert flow == pytest.approx(0.906, abs=0.15)


def test_invariant_density_and_ulam_chain_of_the_logistic_map():
    """Test that a long run and the Ulam chain on 100 cells both match the arcsine density in total variation."""
    edges = np.linspace(0, 1, 101)
    exact = np.diff(2 / np.pi * np.arcsin(np.sqrt(edges)))
    values = np.empty(1_000_000)
    value = 0.3
    for index in range(values.size):
        value = 4 * value * (1 - value)
        values[index] = value
    _, density = invariant_density(values, edges)
    chain = ulam_chain(lambda x: 4 * x * (1 - x), edges, 1000)
    assert 0.5 * np.sum(np.abs(density * 0.01 - exact)) < 0.05
    assert 0.5 * np.sum(np.abs(chain.density * 0.01 - exact)) < 0.05
    np.testing.assert_allclose(chain.matrix.sum(axis=1), 1)
