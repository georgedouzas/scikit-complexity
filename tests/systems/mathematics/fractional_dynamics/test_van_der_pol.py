"""Test the `_van_der_pol` module of `skcomplex.systems.mathematics.fractional_dynamics`."""

import numpy as np
import pytest

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.mathematics.fractional_dynamics import VanDerPolElement


def test_van_der_pol_element_reaches_a_cycle_of_amplitude_two():
    """Test that the Van der Pol oscillator of order 1 started near rest grows to its cycle of amplitude near 2."""
    system = System(elements=[VanDerPolElement('oscillator', x=0.1, mu=1.0)])
    trajectory = Simulation(system, dt=0.1).prepare().run(until=60.0).trajectory_
    late = trajectory.values('oscillator.x')[-200:]
    assert np.max(np.abs(late)) == pytest.approx(2.0, abs=0.1)
