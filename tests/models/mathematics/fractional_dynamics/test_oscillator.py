"""Test the `_oscillator` module of `skcomplex.models.mathematics.fractional_dynamics`."""

import numpy as np

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.mathematics.fractional_dynamics import OscillatorElement


def test_oscillator_element_follows_the_cosine_without_damping():
    """Test that the undamped oscillator of order 1 follows `cos(omega t)`."""
    system = System(elements=[OscillatorElement('oscillator', omega=2.0)])
    trajectory = Simulation(system, dt=0.5).check().prepare().run(until=5.0).trajectory_
    np.testing.assert_allclose(trajectory.values('oscillator.x'), np.cos(2.0 * trajectory.times), atol=1e-4)
