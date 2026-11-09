"""Test the `_pedestrians` module of `skcomplex.systems.social.crowds`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.social.crowds import SocialForceSystem


def test_social_force_walkers_reach_desired_speed_between_walls():
    """Test that both crowds walk at nearly their desired speed in opposite directions and stay between the walls."""
    trajectory = simulate(SocialForceSystem().prepare().system_, until=20, dt=0.2, solver=RK4Solver(dt=0.05))
    assert trajectory.get_values('rightward.x_d1')[-1][:, 0].mean() > 1.0
    assert trajectory.get_values('leftward.x_d1')[-1][:, 0].mean() < -1.0
    heights = np.concatenate(
        [trajectory.get_values('rightward.x')[:, :, 1], trajectory.get_values('leftward.x')[:, :, 1]],
    )
    assert heights.min() > 0
    assert heights.max() < 10
