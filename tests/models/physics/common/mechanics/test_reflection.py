"""Test the `_reflection` module of `skcomplex.models.physics.common.mechanics`."""

import pytest

from skcomplex.core.elements import PointElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.models.physics.common.mechanics import ReflectionInteraction


def test_reflection_scales_the_normal_velocity_and_keeps_the_tangential_one():
    """Test that a reflection off a tilted wall reverses the normal velocity times `e` and keeps the tangential one."""
    wall = ReflectionInteraction(
        'wall',
        target='ball',
        guard=lambda s: s.target.x[0] + s.target.x[1] - 1,
        direction=1,
        restitution=0.5,
    )
    ball = PointElement('ball', x=[0.0, 0.0], x_d1=[1.0, 0.0])
    system = System(elements=[ball], interactions=[wall], space=ContinuousSpace(n_dim=2))
    trajectory = Simulation(system, dt=0.1).check().prepare().run(until=2.0).trajectory_
    velocity = trajectory.values('ball.x_d1')[-1]
    assert velocity.tolist() == pytest.approx([0.25, -0.75])


def test_reflection_checks_one_velocity_per_position():
    """Test that a reflection with more positions than velocities raises an error."""
    wall = ReflectionInteraction('wall', target='ball', guard=lambda s: s.target.x, positions=('x', 'y'))
    with pytest.raises(ValidationError, match='one velocity per position'):
        wall.check()
