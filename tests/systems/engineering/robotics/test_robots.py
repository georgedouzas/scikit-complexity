"""Test the `_robots` module of `skcomplex.systems.engineering.robotics`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.systems.engineering.robotics import LightSeekingRobotsSystem, SwarmtaxisSystem


def test_light_seeking_robots_avoid_light_with_delay():
    """Test that robots with a positive sensory delay spend less time in the light than robots without one."""
    intensities = []
    for delay in (0.0, 5.0):
        system = LightSeekingRobotsSystem(40, delay=delay).prepare().system_
        trajectory = simulate(system, until=150, dt=0.5, solver=EulerMaruyamaSolver(dt=0.01), random_state=1)
        x, y = trajectory.get_values('robots.x'), trajectory.get_values('robots.y')
        intensities.append(np.exp(-(x**2 + y**2))[trajectory.times > 50].mean())
    assert intensities[1] < 0.9 * intensities[0]


def test_light_seeking_robots_stay_in_the_well():
    """Test that the wall keeps every robot inside the well."""
    trajectory = simulate(LightSeekingRobotsSystem(10).prepare().system_, until=20, dt=1.0, random_state=0)
    assert np.hypot(trajectory.get_values('robots.x'), trajectory.get_values('robots.y')).max() <= 2.0 + 1e-9


def test_swarmtaxis_moves_towards_the_beacon():
    """Test that the centre of the swarm moves towards the beacon and the swarm stays together."""
    trajectory = simulate(SwarmtaxisSystem().prepare().system_, until=300, random_state=0)
    start = np.asarray(trajectory.get_agents('robots', 0).locations).mean(axis=0)
    end = np.asarray(trajectory.get_agents('robots', 300).locations)
    assert end.mean(axis=0)[0] - start[0] > 1.0
    assert end.std(axis=0).max() < 10.0
