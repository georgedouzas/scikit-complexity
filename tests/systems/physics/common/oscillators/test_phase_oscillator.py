"""Test the `_phase_oscillator` module of `skcomplex.systems.physics.common.oscillators`."""

import pytest

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.physics.common.oscillators import PhaseOscillatorElement


def test_phase_turns_at_the_natural_frequency():
    """Test that a free phase grows linearly at its natural frequency."""
    system = System(elements=[PhaseOscillatorElement('clock', theta=0.5, omega=-1.5)])
    trajectory = Simulation(system, dt=0.5).prepare().run(until=2.0).trajectory_
    assert trajectory.values('clock.theta')[-1] == pytest.approx(0.5 - 3.0)
