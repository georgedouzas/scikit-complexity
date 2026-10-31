"""Test the `_orbits` module of `skcomplex.models.physics.celestial_mechanics`."""

import math

import numpy as np
import pytest

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
    period = 2 * math.pi * math.sqrt(axis**3 / (G_ * (mass + 1e-6)))
    trajectory = _run(two_body_orbit(eccentricity, G=G_, mass=mass, radius=radius), 10 * period, period / 100)
    star, planet = _series(trajectory, 'star.x', 'planet.x')
    star_v, planet_v = _series(trajectory, 'star.x_d1', 'planet.x_d1')
    relative, velocity = planet - star, planet_v - star_v
    distance = np.linalg.norm(relative, axis=1)
    energy = 0.5 * mass * (star_v**2).sum(1) + 0.5 * 1e-6 * (planet_v**2).sum(1) - G_ * mass * 1e-6 / distance
    momentum = sum(
        weight * (position[:, 0] * speed[:, 1] - position[:, 1] * speed[:, 0])
        for weight, position, speed in ((mass, star, star_v), (1e-6, planet, planet_v))
    )
    assert _relative_spread(energy) < 1e-7
    assert _relative_spread(momentum) < 1e-7
    crossings = _upward_crossings(trajectory.times, relative[:, 1])
    assert len(crossings) >= 9
    assert np.diff(crossings).mean() == pytest.approx(period, rel=1e-3)
    assert np.ptp(np.linalg.norm(velocity, axis=1)) > 0


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
    x, y, vx, vy = _series(trajectory, 'body.x', 'body.y', 'body.x_d1', 'body.y_d1')
    jacobi = x**2 + y**2 + 2 * (1 - mu) / np.hypot(x + mu, y) + 2 * mu / np.hypot(x - 1 + mu, y) - vx**2 - vy**2
    assert np.ptp(jacobi) < 1e-8


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
