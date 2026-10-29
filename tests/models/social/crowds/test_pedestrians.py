"""Test the `_pedestrians` module of `skcomplex.models.social.crowds`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.models.social.crowds import social_force


def test_social_force_walkers_reach_desired_speed_between_walls():
    """Test that both crowds walk at nearly their desired speed in opposite directions and stay between the walls."""
    trajectory = Simulation(social_force(), dt=0.2, solver=RK4Solver(dt=0.05)).check().run(until=20).trajectory_
    assert trajectory.values('rightward.v')[-1][:, 0].mean() > 1.0
    assert trajectory.values('leftward.v')[-1][:, 0].mean() < -1.0
    heights = np.concatenate([trajectory.values('rightward.x')[:, :, 1], trajectory.values('leftward.x')[:, :, 1]])
    assert heights.min() > 0
    assert heights.max() < 10
