"""Test the `_light_emitting_robots` module of `skcomplex.systems.engineering.robotics`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.robotics import LightEmittingRobotsSystem


def _mean_light(delay):
    """Return the light the robots measure, averaged over the robots and the second half of a run of 600 steps."""
    trajectory = simulate(LightEmittingRobotsSystem(n_robots=20, delay=delay), until=600, random_state=0)
    agents = trajectory.get_agents
    return np.mean([agents('robots', frame).values['light'].mean() for frame in range(300, 601, 10)])


def test_light_emitting_robots_cluster_with_a_delay_and_spread_with_a_prediction():
    """Test that robots sensing the light 10 s late gather in brighter places than robots predicting it 10 s ahead."""
    late, now, early = (_mean_light(delay) for delay in (10.0, 0.0, -10.0))
    assert late > now > early
    assert late > 1.5 * early


def test_light_emitting_robots_stay_in_the_periodic_arena():
    """Test that the robots stay inside the arena and move at speeds between the speeds in light and in the dark."""
    trajectory = simulate(LightEmittingRobotsSystem(), until=50)
    snapshot = trajectory.get_agents('robots', 50)
    positions = np.asarray(snapshot.locations)
    assert positions.min() >= 0
    assert positions.max() <= 10
    speeds = snapshot.values['speed']
    assert speeds.min() >= 0.1
    assert speeds.max() <= 1.0
