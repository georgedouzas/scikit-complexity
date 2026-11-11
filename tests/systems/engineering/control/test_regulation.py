"""Test the `_regulation` module of `skcomplex.systems.engineering.control`."""


import numpy as np
import pytest

from skcomplex.core.analysis import linearization
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.control import (
    CartPendulumElement,
    LqrOscillatorSystem,
    PidInvertedPendulumSystem,
)


def test_pid_inverted_pendulum_returns_upright():
    """Test that the PID controller brings the pendulum from a tilt of 0.1 to within 0.01 of upright within 5 s."""
    trajectory = simulate(PidInvertedPendulumSystem())
    assert trajectory.get_values('cart.theta')[0] == pytest.approx(0.1)
    late = trajectory.times >= 2.0
    assert np.all(np.abs(trajectory.get_values('cart.theta')[late]) < 0.01)
    assert np.all(np.abs(trajectory.get_values('tilt')) <= 0.1 + 1e-12)


def test_lqr_oscillator_comes_to_rest():
    """Test that the regulated oscillator loses its energy, unlike the free oscillator, which keeps it."""
    trajectory = simulate(LqrOscillatorSystem())
    assert trajectory.get_values('energy')[0] == pytest.approx(0.5)
    assert trajectory.get_values('energy')[-1] < 1e-4
    free = simulate(LqrOscillatorSystem(r=1e6))
    assert free.get_values('energy')[-1] > 0.4


def test_lqr_oscillator_is_controllable():
    """Test that the oscillator with its force as control is controllable."""
    plant = CompositeSystem(elements=list(LqrOscillatorSystem().prepare().system_.elements))
    assert linearization(plant).controllability == 2


def test_pendulum_without_control_falls():
    """Test that the pendulum without its controller falls away from upright."""
    trajectory = simulate(CompositeSystem(elements=[CartPendulumElement('cart', theta=0.1)]), until=1.0, dt=0.1)
    assert trajectory.get_values('cart.theta')[-1] > 0.5
