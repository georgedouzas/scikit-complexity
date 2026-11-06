"""Test the `_spin_orbit` module of `skcomplex.systems.physics.celestial_mechanics`."""

import math

import numpy as np
import pytest

from skcomplex.systems import create_simulation
from skcomplex.systems.physics.celestial_mechanics import keplerian_dumbbell, relativistic_orbit

from ..conftest import _run, _series, _upward_crossings


def test_keplerian_dumbbell_conserves_energy_and_angular_momentum():
    """Test that the spinning dumbbell keeps its energy and its angular momentum about the normal to its orbit."""
    trajectory = create_simulation('keplerian_dumbbell').prepare().run(until=30.0).trajectory_
    energy, momentum = _series(trajectory, 'dumbbell.energy', 'dumbbell.angular_momentum')
    assert momentum[0] == pytest.approx(1 + 0.25 / 0.9 * 0.01 * 3)
    assert np.ptp(energy) < 1e-8
    assert np.ptp(momentum) < 1e-8
    assert np.abs(_series(trajectory, 'dumbbell.v')[0]).max() < 1e-12


def test_short_keplerian_dumbbell_follows_a_kepler_orbit():
    """Test that a very short dumbbell on the unit circle orbits with the Kepler period `2π` to the order `epsilon²`."""
    trajectory = _run(keplerian_dumbbell(epsilon=1e-3, spin=1.0), 2 * math.pi, 0.01)
    u, theta = _series(trajectory, 'dumbbell.u', 'dumbbell.theta')
    assert np.ptp(u) < 1e-5
    assert theta[-1] == pytest.approx(2 * math.pi, rel=1e-5)


def test_relativistic_orbit_circles_at_the_stable_root_and_precesses_around_it():
    """Test that the smaller root is a circular orbit and that orbits near it advance by `2π / sqrt(1 - 2 ε u)`."""
    alpha, epsilon = 1.0, 0.05
    root = (1 - math.sqrt(1 - 4 * alpha * epsilon)) / (2 * epsilon)
    circle = _series(_run(relativistic_orbit(u=root, alpha=alpha, epsilon=epsilon), 20.0, 0.05), 'planet.u')[0]
    assert np.ptp(circle) < 1e-9
    trajectory = _run(relativistic_orbit(u=root + 1e-3, alpha=alpha, epsilon=epsilon), 60.0, 0.01)
    u = _series(trajectory, 'planet.u')[0]
    crossings = _upward_crossings(trajectory.times, u - root)
    assert np.diff(crossings).mean() == pytest.approx(2 * math.pi / math.sqrt(1 - 2 * epsilon * root), rel=1e-4)


def test_relativistic_orbit_falls_in_beyond_the_unstable_root():
    """Test that a planet started just inside the unstable circular orbit falls towards the centre."""
    alpha, epsilon = 1.0, 0.05
    root = (1 + math.sqrt(1 - 4 * alpha * epsilon)) / (2 * epsilon)
    u = _series(_run(relativistic_orbit(u=root * 1.1, alpha=alpha, epsilon=epsilon), 3.0, 0.01), 'planet.u')[0]
    assert u[-1] > 2 * root
