"""Test the `_pedestrians` module of `skcomplex.models.social.crowds`."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.models.social.crowds import social_force


def test_social_force_walkers_reach_desired_speed_between_walls():
    """Test that both crowds walk at nearly their desired speed in opposite directions and stay between the walls."""
    trajectory = Simulation(social_force(), dt=0.2, solver=RK4(dt=0.05)).run(until=20).trajectory_
    assert trajectory['rightward.v'][-1][:, 0].mean() > 1.0
    assert trajectory['leftward.v'][-1][:, 0].mean() < -1.0
    heights = np.concatenate([trajectory['rightward.x'][:, :, 1], trajectory['leftward.x'][:, :, 1]])
    assert heights.min() > 0 and heights.max() < 10  # noqa: PT018
