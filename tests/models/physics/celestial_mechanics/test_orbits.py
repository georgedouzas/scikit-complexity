"""Test the `_orbits` module of `skcomplex.models.physics.celestial_mechanics`."""

import math

import numpy as np
import pytest

from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import Yoshida4Solver
from skcomplex.errors import SolverError
from skcomplex.models import run
from skcomplex.models.physics.celestial_mechanics import (
    n_body,
    restricted_three_body,
    sitnikov,
    two_body_orbit,
)

from ..conftest import _relative_spread, _run, _series, _upward_crossings


def test_kepler_orbit_conserves_energy_and_momentum_and_follows_the_third_law():
    """Test that the Kepler orbit keeps its energy and angular momentum for ten periods with Kepler's period."""
    eccentricity, G_, mass, radius = 0.5, 1.0, 1000.0, 10.0
    axis = radius / (1 - eccentricity)
    period = 2 * math.pi * math.sqrt(axis**3 / (G_ * mass))
    system = two_body_orbit(eccentricity, G=G_, mass=mass, radius=radius)
    trajectory = _run(system, 10 * period, period / 100, solver=Yoshida4Solver(dt=period / 4000))
    x, y, px, py = _series(trajectory, 'planet.x', 'planet.y', 'planet.px', 'planet.py')
    energy = (px**2 + py**2) / 2 - G_ * mass / np.hypot(x, y)
    momentum = x * py - y * px
    assert _relative_spread(energy) < 1e-8
    assert _relative_spread(momentum) < 1e-8
    assert np.hypot(x[-1] - x[0], y[-1] - y[0]) / radius < 1e-6
    crossings = _upward_crossings(trajectory.times, y)
    assert len(crossings) >= 9
    assert np.diff(crossings).mean() == pytest.approx(period, rel=1e-6)


def test_kepler_orbit_records_its_energy_within_its_invariant_tolerance():
    """Test that the default run of the Kepler problem records its energy and passes its invariant check."""
    trajectory = run('two_body_orbit').trajectory_
    energy = np.asarray(trajectory.values('planet.energy'), dtype=float)
    assert np.ptp(energy) < 1e-6


def test_figure_eight_returns_after_its_period():
    """Test that the three bodies of the figure eight come back to their places after one period."""
    trajectory = _run(n_body(), 6.32591398, 6.32591398 / 100)
    for index in range(3):
        position = _series(trajectory, f'body_{index}.x')[0]
        assert np.allclose(position[-1], position[0], atol=1e-4)
    momentum = sum(_series(trajectory, f'body_{index}.x_d1')[0] for index in range(3))
    assert np.abs(momentum).max() < 1e-8


def test_inner_solar_system_keeps_the_earth_on_its_orbit():
    """Test that the Earth of the inner solar system preset goes around the Sun once a year."""
    trajectory = _run(n_body('inner_solar_system'), 1.0, 0.01)
    sun, earth = _series(trajectory, 'body_0.x', 'body_3.x')
    assert np.linalg.norm(earth - sun, axis=1) == pytest.approx(1.0, rel=1e-3)
    assert np.allclose(earth[-1] - sun[-1], [1.0, 0.0], atol=0.01)


def test_restricted_three_body_conserves_the_jacobi_constant():
    """Test that the light body in the rotating frame keeps its Jacobi constant."""
    mu = 0.01215
    trajectory = _run(restricted_three_body(mu=mu), 20.0, 0.02)
    x, y, px, py = _series(trajectory, 'body.x', 'body.y', 'body.px', 'body.py')
    vx, vy = px + y, py - x
    jacobi = x**2 + y**2 + 2 * (1 - mu) / np.hypot(x + mu, y) + 2 * mu / np.hypot(x - 1 + mu, y) - vx**2 - vy**2
    assert np.ptp(jacobi) < 1e-8


def test_restricted_three_body_records_its_hamiltonian_within_its_invariant_tolerance():
    """Test that the default run of the restricted three-body problem passes its invariant check."""
    trajectory = run('restricted_three_body').trajectory_
    assert np.ptp(np.asarray(trajectory.values('body.energy'), dtype=float)) < 1e-6


def test_restricted_three_body_admits_no_split_of_the_yoshida_solver():
    """Test that the Coriolis terms of the rotating frame make the Yoshida solver name a momentum and RK4Solver."""
    with pytest.raises(SolverError, match=r'rate of body\.p[xy] reads .*body\.p[xy].*RK4Solver'):
        Yoshida4Solver().check().prepare(compile_system(restricted_three_body()))


def test_restricted_three_body_defaults_are_chaotic():
    """Test that the default light body visits the Moon and separates from a neighbour started 1e-8 away."""
    mu = 0.01215
    first = _run(restricted_three_body(), 20.0, 0.02)
    second = _run(restricted_three_body(x=0.5 + 1e-8), 20.0, 0.02)
    x, y = _series(first, 'body.x', 'body.y')
    other_x, other_y = _series(second, 'body.x', 'body.y')
    assert np.hypot(x - 1 + mu, y).min() < 0.2
    assert np.hypot(x - other_x, y - other_y)[-1] > 1e-6


def test_sitnikov_conserves_energy_with_circular_primaries_and_stays_symmetric():
    """Test that the Sitnikov body conserves its energy above circular primaries and oscillates symmetrically."""
    trajectory = _run(sitnikov(0.5), 30.0, 0.05)
    z, speed = _series(trajectory, 'body.z', 'body.z_d1')
    energy = speed**2 / 2 - 1 / np.sqrt(z**2 + 0.25)
    assert np.abs(energy - energy[0]).max() < 1e-6
    assert z.min() == pytest.approx(-0.5, abs=1e-4)
