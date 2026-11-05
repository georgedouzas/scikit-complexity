"""Test the `_regulation` module of `skcomplex.systems.engineering.control`."""

import numpy as np
import pytest

from skcomplex.core.analysis import LinearizationAnalysis
from skcomplex.core.components.system import System
from skcomplex.systems import create_simulation


def test_pid_inverted_pendulum_returns_upright():
    """Test that the PID controller brings the pendulum from a tilt of 0.1 to within 0.01 of upright within 5 s."""
    trajectory = create_simulation('pid_inverted_pendulum').prepare().run().trajectory_
    assert trajectory.values('cart.theta')[0] == pytest.approx(0.1)
    late = trajectory.times >= 2.0
    assert np.all(np.abs(trajectory.values('cart.theta')[late]) < 0.01)
    assert np.all(np.abs(trajectory.values('tilt')) <= 0.1 + 1e-12)


def test_lqr_oscillator_comes_to_rest():
    """Test that the regulated oscillator loses its energy, unlike the free oscillator, which keeps it."""
    trajectory = create_simulation('lqr_oscillator').prepare().run().trajectory_
    assert trajectory.values('energy')[0] == pytest.approx(0.5)
    assert trajectory.values('energy')[-1] < 1e-4
    free = create_simulation('lqr_oscillator', r=1e6).prepare().run().trajectory_
    assert free.values('energy')[-1] > 0.4


def test_lqr_oscillator_is_controllable():
    """Test that the oscillator with its force as control is controllable."""
    plant = create_simulation('lqr_oscillator').prepare().run(until=0.1).system
    plant = System(elements=list(plant.elements))
    assert LinearizationAnalysis().prepare(plant).run().controllability_ == 2
