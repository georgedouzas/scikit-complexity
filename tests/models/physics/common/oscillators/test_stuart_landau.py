"""Test the `_stuart_landau` module of `skcomplex.models.physics.common.oscillators`."""

import math

import numpy as np
import pytest

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.physics.common.oscillators import StuartLandauElement


def _run(**parameters):
    """Return the trajectory of one Stuart-Landau oscillator."""
    system = System(elements=[StuartLandauElement('z', **parameters)])
    solver = ScipyODESolver(rtol=1e-9, atol=1e-11)
    return Simulation(system, dt=0.1, solver=solver).check().prepare().run(until=40.0).trajectory_


@pytest.mark.parametrize(('mu', 'kappa'), [(1.0, 1.0), (2.0, 0.5), (-0.5, 1.0)])
def test_amplitude_settles_at_the_limit_cycle(mu, kappa):
    """Test that the amplitude settles at `sqrt(mu / kappa)` for `mu > 0` and at 0 for `mu < 0`."""
    trajectory = _run(x=0.3, mu=mu, kappa=kappa)
    radius = math.hypot(trajectory.values('z.x')[-1], trajectory.values('z.y')[-1])
    assert radius == pytest.approx(math.sqrt(max(mu, 0) / kappa), abs=1e-6)


def test_twist_and_time_scale_set_the_frequency():
    """Test that the oscillator turns on its cycle at the frequency `tau (omega - beta mu)`."""
    trajectory = _run(x=1.0, mu=1.0, omega=2.0, beta=0.5, tau=2.0)
    angle = np.unwrap(np.arctan2(trajectory.values('z.y'), trajectory.values('z.x')))
    assert angle[-1] - angle[0] == pytest.approx(2.0 * (2.0 - 0.5) * 40.0, rel=1e-6)
