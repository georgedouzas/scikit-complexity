"""Test the `_flocks` module of `skcomplex.models.biology.collective_behaviour`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.collective_behaviour import (
    bird_swarm,
    boids,
    boids_3d,
)


def test_boids_align():
    """Test that boids starting with random headings become a polarized flock."""
    simulation = Simulation(boids(), dt=1, random_state=0).check().prepare().run(until=100)

    def polarization(frame):
        snapshot = simulation.trajectory_.agents('flock', frame)
        velocity = np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)
        return np.hypot(*(velocity / np.hypot(velocity[:, :1], velocity[:, 1:])).mean(axis=0))

    assert polarization(0) < 0.3
    assert polarization(100) > 0.7


def test_boids_3d_align_in_three_dimensions():
    """Test that boids with only the alignment rule polarize and keep their spread over the depth of the box."""
    system = boids_3d(50, cohesion=0.0, separation=0.0, alignment=0.1, random_state=0)
    trajectory = Simulation(system, dt=1, random_state=0).check().prepare().run(until=200).trajectory_

    def polarization(frame):
        velocity = trajectory.agents('flock', frame).values['velocity']
        return np.linalg.norm((velocity / np.linalg.norm(velocity, axis=1, keepdims=True)).mean(axis=0))

    assert polarization(0) < 0.3
    assert polarization(200) > 0.9
    depths = trajectory.agents('flock', 200).locations[:, 2]
    assert trajectory.agents('flock', 200).locations.shape == (50, 3)
    assert np.ptp(depths) > system.space.extent[2] / 2


def test_bird_swarm_flies_together():
    """Test that the swarm reaches the speed `sqrt(alpha / beta)` and stays together."""
    trajectory = Simulation(bird_swarm(), dt=0.5).check().prepare().run(until=50).trajectory_
    positions, velocities = trajectory.values('birds.x')[-1], trajectory.values('birds.x_d1')[-1]
    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    assert np.median(speeds) == pytest.approx(math.sqrt(2.0), rel=0.1)
    assert np.hypot(*(positions - positions.mean(axis=0)).T).max() < 10.0
