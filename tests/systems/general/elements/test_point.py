"""Test the `_point` module of `skcomplex.systems.general.elements`."""

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import VisualSpec
from skcomplex.systems.general.elements import PointElement


def test_point_moves_with_its_first_derivative():
    """Test that a free point moves along its first derivative."""
    system = CompositeSystem(elements=[PointElement('ball', x_d1=[1.0, 0.5])], space=ContinuousSpace(n_dim=2))
    assert simulate(system, until=2.0, dt=1.0).get_values('ball.x')[-1].tolist() == [2.0, 1.0]


def test_point_declares_a_vector_position_of_order_two():
    """Test that the state `x` of a point is a vector of order 2 with the role of the position."""
    state = PointElement('ball').get_states()['x']
    assert (state.vector, state.order, state.role) == (True, 2, 'position')
    assert PointElement('ball').prepare().default_visual_ == VisualSpec('points', {'position': 'x'})
