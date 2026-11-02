"""Test the `_cart_pendulum` module of `skcomplex.models.engineering.control`."""

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.engineering.control import CartPendulumElement


def test_pendulum_without_control_falls():
    """Test that the pendulum without its controller falls away from upright."""
    trajectory = (
        Simulation(System(elements=[CartPendulumElement('cart', theta=0.1)]), dt=0.1)
        .check()
        .prepare()
        .run(until=1.0)
        .trajectory_
    )
    assert trajectory.values('cart.theta')[-1] > 0.5
