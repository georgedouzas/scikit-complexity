"""Test the `_stuart_landau` module of `skcomplex.core.elements`."""

import numpy as np
import pytest

from skcomplex.core.elements import GroupElement, StuartLandauElement
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


def _run(element, until=40.0, dt=0.01):
    """Return the real and imaginary parts of the amplitude of one oscillator."""
    trajectory = Simulation(System(elements=[element]), dt=dt).check().prepare().run(until=until).trajectory_
    return np.asarray(trajectory.values('z.x')), np.asarray(trajectory.values('z.y'))


@pytest.mark.parametrize(('mu', 'a'), [(1.0, 1.0), (4.0, 1.0), (2.0, 0.5)])
def test_stuart_landau_settles_on_its_limit_cycle_radius(mu, a):
    """Test that the amplitude settles on the limit-cycle radius `sqrt(mu / a)`."""
    x, y = _run(StuartLandauElement('z', x=0.1, mu=mu, a=a))
    assert np.hypot(x[-1], y[-1]) == pytest.approx(np.sqrt(mu / a), rel=1e-3)


def test_stuart_landau_turns_at_its_nonlinear_frequency():
    """Test that the phase on the limit cycle turns at the angular frequency `omega - beta mu / a`."""
    x, y = _run(StuartLandauElement('z', x=1.0, mu=1.0, omega=2.0, a=1.0, beta=0.5), until=20.0)
    phase = np.unwrap(np.arctan2(y, x))
    assert (phase[-1] - phase[0]) / 20.0 == pytest.approx(1.5, rel=1e-3)


def test_stuart_landau_decays_below_the_bifurcation():
    """Test that the amplitude decays to the origin for a negative growth rate."""
    x, y = _run(StuartLandauElement('z', x=1.0, mu=-1.0))
    assert np.hypot(x[-1], y[-1]) < 1e-6


def test_stuart_landau_runs_as_a_group():
    """Test that a group of oscillators with different growth rates settles on their own radii."""
    group = GroupElement('z', element=StuartLandauElement, size=2, values={'x': 0.1, 'mu': np.array([1.0, 9.0])})
    trajectory = Simulation(System(elements=[group]), dt=0.01).check().prepare().run(until=40).trajectory_
    x, y = np.asarray(trajectory.values('z.x'))[-1], np.asarray(trajectory.values('z.y'))[-1]
    np.testing.assert_allclose(np.hypot(x, y), [1.0, 3.0], rtol=1e-3)
