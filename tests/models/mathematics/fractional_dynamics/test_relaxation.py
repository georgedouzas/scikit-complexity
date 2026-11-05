"""Test the `_relaxation` module of `skcomplex.models.mathematics.fractional_dynamics`."""

import numpy as np

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.mathematics.fractional_dynamics import RelaxationElement


def test_relaxation_element_decays_exponentially():
    """Test that relaxation of order 1 decays as `exp(-lam t)`."""
    system = System(elements=[RelaxationElement('relaxation', x=2.0, lam=0.5)])
    trajectory = Simulation(system, dt=1.0).prepare().run(until=4.0).trajectory_
    np.testing.assert_allclose(trajectory.values('relaxation.x'), 2.0 * np.exp(-0.5 * trajectory.times), rtol=1e-4)
