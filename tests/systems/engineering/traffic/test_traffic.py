"""Test the `_traffic` module of `skcomplex.systems.engineering.traffic`."""

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.engineering.traffic import NagelSchreckenbergSystem


def test_traffic_jams_lower_flow_and_move_backward():
    """Test that dense traffic flows less than near the critical density and its jams move backward."""
    flows, velocities = {}, {}
    for density in (0.15, 0.5):
        trajectory = (
            Simulation(NagelSchreckenbergSystem(1000, density=density).prepare().system_, dt=1, random_state=0)
            .prepare()
            .run(600)
            .trajectory_
        )
        velocities[density] = trajectory.values('road.velocity')[100:]
        flows[density] = np.where(velocities[density] >= 0, velocities[density], 0).sum(axis=1).mean() / 1000
    assert flows[0.5] < flows[0.15]
    stopped = (velocities[0.5] == 0).astype(float)
    backward = np.mean(stopped[1:] * np.roll(stopped[:-1], -1, axis=1))
    forward = np.mean(stopped[1:] * np.roll(stopped[:-1], 1, axis=1))
    assert backward > 1.5 * forward


def test_traffic_conserves_cars():
    """Test that the number of cars on the ring never changes."""
    trajectory = (
        Simulation(NagelSchreckenbergSystem(200, density=0.3).prepare().system_, dt=1, random_state=1)
        .prepare()
        .run(100)
        .trajectory_
    )
    assert np.all((trajectory.values('road.velocity') >= 0).sum(axis=1) == 60)
