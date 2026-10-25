"""Test the `_regulation` module of `skcomplex.models.engineering.control`."""

import numpy as np
import pytest

from skcomplex.core.system import System
from skcomplex.engine.analysis import controllability, linearize
from skcomplex.engine.runtime import Simulation
from skcomplex.models import run
from skcomplex.models.engineering.control import CartPendulumComponent


def test_pid_inverted_pendulum_returns_upright():
    """Test that the PID controller brings the pendulum from a tilt of 0.1 to within 0.01 of upright within 5 s."""
    trajectory = run('pid_inverted_pendulum').trajectory_
    assert trajectory['cart.theta'][0] == pytest.approx(0.1)
    late = trajectory.times >= 2.0
    assert np.all(np.abs(trajectory['cart.theta'][late]) < 0.01)
    assert np.all(np.abs(trajectory['tilt']) <= 0.1 + 1e-12)


def test_pendulum_without_control_falls():
    """Test that the pendulum without its controller falls away from upright."""
    trajectory = (
        Simulation(System(components=[CartPendulumComponent('cart', theta=0.1)]), dt=0.1).run(until=1.0).trajectory_
    )
    assert trajectory['cart.theta'][-1] > 0.5


def test_lqr_oscillator_comes_to_rest():
    """Test that the regulated oscillator loses its energy, unlike the free oscillator, which keeps it."""
    trajectory = run('lqr_oscillator').trajectory_
    assert trajectory['energy'][0] == pytest.approx(0.5)
    assert trajectory['energy'][-1] < 1e-4
    free = run('lqr_oscillator', r=1e6).trajectory_
    assert free['energy'][-1] > 0.4


def test_lqr_oscillator_is_controllable():
    """Test that the oscillator with its force as control is controllable."""
    plant = run('lqr_oscillator', settings={'until': 0.1}).system
    plant = System(components=list(plant.components))
    model = linearize(plant)
    assert controllability(model.A, model.B) == 2
