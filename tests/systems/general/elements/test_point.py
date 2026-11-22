"""Test the `_point` module of `skcomplex.systems.general.elements`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import vector

from skcomplex.core.components.elements import VisualSpec
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System
from skcomplex.systems.general.elements import PointElement


def _flight(**members: object) -> System:
    """Return points under a uniform pull of 1 along the second axis, the acceleration `x'' = (0, -1)`."""
    gravity = Interaction(
        'gravity',
        roles={'target': 'ball'},
        laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), vector([0, -1]))],
    )
    return System(elements=[PointElement('ball', **members)], interactions=[gravity], space=ContinuousSpace(n_dim=2))


def test_point_moves_by_its_acceleration_from_its_starting_velocity():
    """Test that a point leaves with its velocity `x'` and falls along the parabola of its acceleration."""
    system = _flight(x=np.array([0.0, 0.0]), values={"x'": np.array([1.0, 2.0])}).prepare()
    trajectory = system.simulate(until=2.0, dt=1.0)
    assert trajectory.get_values('ball.x')[-1] == pytest.approx([2.0, 2.0])
    assert trajectory.get_values("ball.x'")[-1] == pytest.approx([1.0, 0.0])


def test_point_holds_one_position_and_velocity_per_member():
    """Test that a group of points has the position and the velocity `x'` of every member as its states."""
    system = _flight(x=np.zeros((3, 2)), size=3).prepare()
    assert system.states_ == ('ball.x[i, 0]', 'ball.x[i, 1]', "ball.x'[i, 0]", "ball.x'[i, 1]")


def test_point_is_drawn_at_its_position():
    """Test that a point is drawn as a point at its position."""
    assert PointElement.visual == VisualSpec('points', {'position': 'x'})
