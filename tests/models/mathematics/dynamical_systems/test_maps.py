"""Test the `_maps` module of `skcomplex.models.mathematics.dynamical_systems`."""

import math

import numpy as np

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.models import run
from skcomplex.models.mathematics.dynamical_systems import (
    baker_map,
    cat_map,
    circle_map,
    intermittency_map,
    newton_raphson_map,
    standard_map,
    tent_map,
    towel_map,
)

from ..conftest import _run

GOLDEN_MEAN = (math.sqrt(5) - 1) / 2


CRITICAL_KICK = 0.971635


def _momentum_range(K, theta, iterations):
    """Return the spread of the momentum of a standard map orbit started near the golden-mean torus."""
    momentum = _run(standard_map(theta=theta, p=2 * math.pi * GOLDEN_MEAN, K=K), iterations)['map.p']
    return momentum.max() - momentum.min()


def test_standard_map_diffuses_above_the_critical_kick():
    """Test that a standard map orbit diffuses in momentum above K = 0.971635 and stays bounded below it."""
    assert _momentum_range(CRITICAL_KICK + 0.23, 0.0, 20000) > 4 * math.pi
    assert all(_momentum_range(CRITICAL_KICK - 0.02, theta, 10000) < 2 * math.pi for theta in (0.0, math.pi / 2))


def test_circle_map_rotates_rationally_without_nonlinearity():
    """Test that the circle map without nonlinearity is a rotation of period 4 for a quarter turn."""
    theta = _run(circle_map(0.1, Omega=0.25, K=0.0), 8)['map.theta']
    np.testing.assert_allclose(theta[4], theta[0], atol=1e-12)
    np.testing.assert_allclose(theta[1], 0.35, atol=1e-12)


def test_tent_map_decays_below_unit_slope_and_stays_in_its_interval_above():
    """Test that the tent map tends to 0 for a slope below 1 and stays in `[0, μ / 2]` for a chaotic slope."""
    assert _run(tent_map(0.3, mu=0.8), 100)['map.x'][-1] < 1e-8
    x = _run(tent_map(0.3, mu=1.9), 500)['map.x']
    assert x.min() >= 0
    assert x.max() <= 0.95 + 1e-12


def test_baker_map_leaves_a_gap_when_it_contracts():
    """Test that the dissipative baker map leaves the band between its two images empty."""
    y = _run(baker_map(0.2, 0.5, alpha=0.4, lambda_a=0.3, lambda_b=0.3), 300)['map.y'][20:]
    assert not np.any((y > 0.3 + 1e-9) & (y < 0.7 - 1e-9))
    y = _run(baker_map(0.2, 0.5), 300)['map.y'][20:]
    assert np.any((y > 0.4) & (y < 0.6))


def test_towel_map_is_hyperchaotic():
    """Test that Rössler's towel map has two positive Lyapunov exponents."""
    exponents = lyapunov_spectrum(towel_map(), until=5000)
    assert exponents[1] > 0.1
    assert exponents[2] < 0


def test_intermittency_map_has_a_long_laminar_phase():
    """Test that the Pomeau-Manneville map lingers near 0 for about `1 / x0` steps."""
    x = _run(intermittency_map(0.01, z=2.0), 300)['map.x']
    assert 60 < np.argmax(x > 0.1) < 120


def test_newton_raphson_map_finds_the_cube_roots_of_unity():
    """Test that Newton's method converges to the cube root of unity whose basin holds the start."""
    first = _run(newton_raphson_map(1.5, 0.1), 50)
    np.testing.assert_allclose([first['map.x'][-1], first['map.y'][-1]], [1.0, 0.0], atol=1e-10)
    second = _run(newton_raphson_map(-1.0, 1.0), 50)
    np.testing.assert_allclose([second['map.x'][-1], second['map.y'][-1]], [-0.5, math.sqrt(3) / 2], atol=1e-10)


def test_cat_map_has_a_periodic_orbit_of_period_three():
    """Test that the cat map sends `(1/2, 1/2)` around an orbit of period 3."""
    trajectory = _run(cat_map(0.5, 0.5), 3)
    np.testing.assert_allclose([trajectory['map.x'][3], trajectory['map.y'][3]], [0.5, 0.5])
    assert trajectory['map.y'][1] == 0.0


def test_henon_map_stays_on_attractor():
    """Test that the Hénon map stays within its attractor's bounds."""
    trajectory = run('henon_map').trajectory_
    assert np.abs(trajectory['map.x'][100:]).max() < 1.5
    assert np.abs(trajectory['map.y'][100:]).max() < 0.5
