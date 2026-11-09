"""Test the `_maps` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import lyapunov
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.mathematics.dynamical_systems import (
    BakerMapSystem,
    CatMapSystem,
    CDNMapSystem,
    CircleMapSystem,
    ConservativeHenonMapSystem,
    FermiUlamMapSystem,
    HenonMapSystem,
    IntermittencyMapSystem,
    LoziMapSystem,
    NewtonRaphsonMapSystem,
    StandardMapSystem,
    TentMapSystem,
    TowelMapSystem,
)

from ..conftest import _run

GOLDEN_MEAN = (math.sqrt(5) - 1) / 2


CRITICAL_KICK = 0.971635


def _momentum_range(K, theta, iterations):
    """Return the spread of the momentum of a standard map orbit started near the golden-mean torus."""
    momentum = _run(StandardMapSystem(theta=theta, p=2 * math.pi * GOLDEN_MEAN, K=K), iterations).get_values('map.p')
    return momentum.max() - momentum.min()


def test_standard_map_diffuses_above_the_critical_kick():
    """Test that a standard map orbit diffuses in momentum above K = 0.971635 and stays bounded below it."""
    assert _momentum_range(CRITICAL_KICK + 0.23, 0.0, 20000) > 4 * math.pi
    assert all(_momentum_range(CRITICAL_KICK - 0.02, theta, 10000) < 2 * math.pi for theta in (0.0, math.pi / 2))


def test_circle_map_rotates_rationally_without_nonlinearity():
    """Test that the circle map without nonlinearity is a rotation of period 4 for a quarter turn."""
    theta = _run(CircleMapSystem(0.1, Omega=0.25, K=0.0), 8).get_values('map.theta')
    np.testing.assert_allclose(theta[4], theta[0], atol=1e-12)
    np.testing.assert_allclose(theta[1], 0.35, atol=1e-12)


def test_tent_map_decays_below_unit_slope_and_stays_in_its_interval_above():
    """Test that the tent map tends to 0 for a slope below 1 and stays in `[0, μ / 2]` for a chaotic slope."""
    assert _run(TentMapSystem(0.3, mu=0.8), 100).get_values('map.x')[-1] < 1e-8
    x = _run(TentMapSystem(0.3, mu=1.9), 500).get_values('map.x')
    assert x.min() >= 0
    assert x.max() <= 0.95 + 1e-12


def test_baker_map_leaves_a_gap_when_it_contracts():
    """Test that the dissipative baker map leaves the band between its two images empty."""
    y = _run(BakerMapSystem(0.2, 0.5, alpha=0.4, lambda_a=0.3, lambda_b=0.3), 300).get_values('map.y')[20:]
    assert not np.any((y > 0.3 + 1e-9) & (y < 0.7 - 1e-9))
    y = _run(BakerMapSystem(0.2, 0.5), 300).get_values('map.y')[20:]
    assert np.any((y > 0.4) & (y < 0.6))


def test_towel_map_is_hyperchaotic():
    """Test that Rössler's towel map has two positive Lyapunov exponents."""
    exponents = lyapunov(TowelMapSystem(), until=5000).exponents
    assert exponents[1] > 0.1
    assert exponents[2] < 0


def test_intermittency_map_has_a_long_laminar_phase():
    """Test that the Pomeau-Manneville map lingers near 0 for about `1 / x0` steps."""
    x = _run(IntermittencyMapSystem(0.01, z=2.0), 300).get_values('map.x')
    assert 60 < np.argmax(x > 0.1) < 120


def test_newton_raphson_map_finds_the_cube_roots_of_unity():
    """Test that Newton's method converges to the cube root of unity whose basin holds the start."""
    first = _run(NewtonRaphsonMapSystem(1.5, 0.1), 50)
    np.testing.assert_allclose([first.get_values('map.x')[-1], first.get_values('map.y')[-1]], [1.0, 0.0], atol=1e-10)
    second = _run(NewtonRaphsonMapSystem(-1.0, 1.0), 50)
    np.testing.assert_allclose(
        [second.get_values('map.x')[-1], second.get_values('map.y')[-1]],
        [-0.5, math.sqrt(3) / 2],
        atol=1e-10,
    )


def test_cat_map_has_a_periodic_orbit_of_period_three():
    """Test that the cat map sends `(1/2, 1/2)` around an orbit of period 3."""
    trajectory = _run(CatMapSystem(0.5, 0.5), 3)
    np.testing.assert_allclose([trajectory.get_values('map.x')[3], trajectory.get_values('map.y')[3]], [0.5, 0.5])
    assert trajectory.get_values('map.y')[1] == 0.0


def test_henon_map_stays_on_attractor():
    """Test that the Hénon map stays within its attractor's bounds."""
    trajectory = simulate(HenonMapSystem(), random_state=0)
    assert np.abs(trajectory.get_values('map.x')[100:]).max() < 1.5
    assert np.abs(trajectory.get_values('map.y')[100:]).max() < 0.5


def test_lozi_map_is_chaotic():
    """Test that the Lozi map has the exponents near 0.47 and -1.16 of its attractor for `a = 1.7`, `b = 0.5`."""
    exponents = lyapunov(LoziMapSystem(), until=10000).exponents
    np.testing.assert_allclose(exponents, [0.47, -1.16], atol=0.03)


def test_conservative_henon_map_keeps_its_saddle_and_its_centre():
    """Test that `(2 tan(π nu), -2 tan²(π nu))` is fixed and orbits near the centre at the origin stay bounded."""
    nu = 0.2114
    tangent = math.tan(math.pi * nu)
    saddle = _run(ConservativeHenonMapSystem(2 * tangent, -2 * tangent**2, nu=nu), 5)
    np.testing.assert_allclose(saddle.get_values('map.x'), 2 * tangent, atol=1e-12)
    np.testing.assert_allclose(saddle.get_values('map.y'), -2 * tangent**2, atol=1e-12)
    orbit = _run(ConservativeHenonMapSystem(0.1, 0.0, nu=nu), 2000)
    assert np.hypot(orbit.get_values('map.x'), orbit.get_values('map.y')).max() < 0.2


def test_conservative_henon_map_preserves_area():
    """Test that the map has two exponents of opposite sign that sum to zero."""
    exponents = lyapunov(ConservativeHenonMapSystem(0.5, 0.0), until=5000).exponents
    assert exponents.sum() == pytest.approx(0.0, abs=1e-9)


def test_fermi_ulam_map_rests_at_its_fixed_points():
    """Test that the speeds `1 / m` at the phases 0 and π are fixed points of the Fermi-Ulam map."""
    for speed, phase in ((0.5, 0.0), (1.0, math.pi), (1 / 3, 0.0)):
        trajectory = _run(FermiUlamMapSystem(speed, phase, epsilon=0.1), 5)
        np.testing.assert_allclose(trajectory.get_values('map.u'), speed, atol=1e-12)
        np.testing.assert_allclose(np.cos(trajectory.get_values('map.theta')), math.cos(phase), atol=1e-9)


def test_cdn_map_is_chaotic_inside_the_window_of_dilao_and_periodic_outside():
    """Test that the Gaussian map at `alpha = 2.8` has a positive exponent at `β = -0.5` and a negative one at -0.1."""
    chaotic = lyapunov(CDNMapSystem(beta=-0.5), until=10000).exponents[0]
    periodic = lyapunov(CDNMapSystem(beta=-0.1), until=10000).exponents[0]
    assert chaotic > 0.3
    assert periodic < -0.5
