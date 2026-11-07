"""Test the `_pedestrians` module of `skcomplex.systems.social.crowds`."""

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.social.crowds import SocialForceSystem


def test_social_force_walkers_reach_desired_speed_between_walls():
    """Test that both crowds walk at nearly their desired speed in opposite directions and stay between the walls."""
    trajectory = (
        Simulation(SocialForceSystem().prepare().system_, dt=0.2, solver=RK4Solver(dt=0.05))
        .prepare()
        .run(until=20)
        .trajectory_
    )
    assert trajectory.values('rightward.x_d1')[-1][:, 0].mean() > 1.0
    assert trajectory.values('leftward.x_d1')[-1][:, 0].mean() < -1.0
    heights = np.concatenate([trajectory.values('rightward.x')[:, :, 1], trajectory.values('leftward.x')[:, :, 1]])
    assert heights.min() > 0
    assert heights.max() < 10
