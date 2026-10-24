"""Test the `_robots` module of `skcomplex.models.engineering.robotics`."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.models.engineering.robotics import light_seeking_robots, swarmtaxis


def test_light_seeking_robots_avoid_light_with_delay():
    """Test that robots with a positive sensory delay spend less time in the light than robots without one."""
    intensities = []
    for delay in (0.0, 5.0):
        system = light_seeking_robots(40, delay=delay)
        trajectory = Simulation(system, dt=0.5, solver=EulerMaruyama(dt=0.01), random_state=1).run(until=150)
        x, y = trajectory.trajectory_['robots.x'], trajectory.trajectory_['robots.y']
        intensities.append(np.exp(-(x**2 + y**2))[trajectory.trajectory_.times > 50].mean())
    assert intensities[1] < 0.9 * intensities[0]


def test_light_seeking_robots_stay_in_the_well():
    """Test that the wall keeps every robot inside the well."""
    trajectory = Simulation(light_seeking_robots(10), dt=1.0, random_state=0).run(until=20).trajectory_
    assert np.hypot(trajectory['robots.x'], trajectory['robots.y']).max() <= 2.0 + 1e-9


def test_swarmtaxis_moves_towards_the_beacon():
    """Test that the centre of the swarm moves towards the beacon and the swarm stays together."""
    trajectory = Simulation(swarmtaxis(), random_state=0).run(until=300).trajectory_
    start = np.asarray(trajectory.agents('robots', 0).locations).mean(axis=0)
    end = np.asarray(trajectory.agents('robots', 300).locations)
    assert end.mean(axis=0)[0] - start[0] > 1.0
    assert end.std(axis=0).max() < 10.0
