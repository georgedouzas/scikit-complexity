"""Test the `_reflection` module of `skcomplex.systems.physics.common.mechanics`."""

import pytest

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.general.elements import PointElement
from skcomplex.systems.physics.common.mechanics import ReflectionInteraction


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
    system = CompositeSystem(elements=[ball], interactions=[wall], space=ContinuousSpace(n_dim=2))
    trajectory = simulate(system, until=2.0, dt=0.1)
    velocity = trajectory.values('ball.x_d1')[-1]
    assert velocity.tolist() == pytest.approx([0.25, -0.75])


def test_reflection_checks_one_velocity_per_position():
    """Test that a reflection with more positions than velocities raises an error."""
    wall = ReflectionInteraction('wall', target='ball', guard=lambda s: s.target.x, positions=('x', 'y'))
    with pytest.raises(ValidationError, match='one velocity per position'):
        wall.prepare()


@pytest.mark.parametrize(('height', 'reflected'), [(0.5, True), (-0.5, False)])
def test_reflection_reflects_only_on_its_region(height, reflected):
    """Test that a wall restricted to a region reflects the point there and lets it cross the rest of the surface."""
    wall = ReflectionInteraction(
        'wall',
        target='ball',
        guard=lambda s: s.target.x[0] - 1,
        direction=1,
        region=lambda s: s.target.x[1] > 0,
    )
    ball = PointElement('ball', x=[0.0, height], x_d1=[1.0, 0.0])
    system = CompositeSystem(elements=[ball], interactions=[wall], space=ContinuousSpace(n_dim=2))
    trajectory = simulate(system, until=2.0, dt=0.1)
    assert trajectory.values('ball.x_d1')[-1].tolist() == pytest.approx([-1.0 if reflected else 1.0, 0.0])


def test_reflection_checks_a_callable_region():
    """Test that a reflection whose region is not a function raises an error."""
    wall = ReflectionInteraction('wall', target='ball', guard=lambda s: s.target.x, region=True)
    with pytest.raises(ValidationError, match='region'):
        wall.prepare()
