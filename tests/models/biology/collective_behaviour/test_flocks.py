"""Test the `_flocks` module of `skcomplex.models.biology.collective_behaviour`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.collective_behaviour import (
    bird_swarm,
    boids,
)


def test_boids_align():
    """Test that boids starting with random headings become a polarized flock."""
    simulation = Simulation(boids(), dt=1, random_state=0).run(until=100)

    def polarization(frame):
        snapshot = simulation.trajectory_.agents('flock', frame)
        velocity = np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)
        return np.hypot(*(velocity / np.hypot(velocity[:, :1], velocity[:, 1:])).mean(axis=0))

    assert polarization(0) < 0.3
    assert polarization(100) > 0.7


def test_bird_swarm_flies_together():
    """Test that the swarm reaches the speed `sqrt(alpha / beta)` and stays together."""
    trajectory = Simulation(bird_swarm(), dt=0.5).run(until=50).trajectory_
    positions, velocities = trajectory['birds.x'][-1], trajectory['birds.v'][-1]
    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    assert np.median(speeds) == pytest.approx(math.sqrt(2.0), rel=0.1)
    assert np.hypot(*(positions - positions.mean(axis=0)).T).max() < 10.0
