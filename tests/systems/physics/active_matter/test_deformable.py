"""Test the `_deformable` module of `skcomplex.systems.physics.active_matter`."""

import math

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.active_matter import DeformableActiveParticlesSystem


def _velocities(kappa):
    """Return the speeds and the unwrapped directions of the velocities over a run at a relaxation rate."""
    trajectory = simulate(DeformableActiveParticlesSystem(kappa=kappa), random_state=0)
    velocity = np.asarray(trajectory.get_values('cells.v'))
    return np.hypot(velocity[..., 0], velocity[..., 1]), np.unwrap(
        np.arctan2(velocity[..., 1], velocity[..., 0]),
        axis=0,
    )


def test_deformable_particles_move_straight_at_the_published_speed_for_fast_relaxation():
    """Test that for `kappa = 2` every particle moves straight at `sqrt(gamma / (1 + a b / (2 kappa)))`."""
    speeds, directions = _velocities(2.0)
    np.testing.assert_allclose(speeds[-1], math.sqrt(1 / 1.25), rtol=1e-6)
    assert np.abs(directions[-1] - directions[-41]).max() < 1e-6


def test_deformable_particles_circle_for_slow_relaxation():
    """Test that for `kappa = 0.5` every particle circles at the speed and angular velocity of the circular solution."""
    speeds, directions = _velocities(0.5)
    np.testing.assert_allclose(speeds[-1], math.sqrt(0.75), rtol=2e-3)
    turning = np.abs(directions[-1] - directions[-41]) / 10.0
    np.testing.assert_allclose(turning, math.sqrt(0.75 - 0.25) / 2, rtol=1e-2)
