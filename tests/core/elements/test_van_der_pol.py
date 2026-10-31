"""Test the `_van_der_pol` module of `skcomplex.core.elements`."""

import numpy as np
import pytest

from skcomplex.core.elements import VanDerPolElement
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


def _displacement(element, until, dt=0.01):
    """Return the times and the displacement of one oscillator."""
    trajectory = Simulation(System(elements=[element]), dt=dt).check().prepare().run(until=until).trajectory_
    return np.asarray(trajectory.times), np.asarray(trajectory.values('osc.x'))


def _period(times, x, transient):
    """Return the mean time between upward zero crossings after a transient."""
    upward = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    crossings = times[upward] - x[upward] * (times[upward + 1] - times[upward]) / (x[upward + 1] - x[upward])
    crossings = crossings[crossings > transient]
    return float(np.diff(crossings).mean())


def test_van_der_pol_settles_on_amplitude_two():
    """Test that a weakly damped oscillator settles on the limit cycle of amplitude 2 and period close to `2 pi`."""
    times, x = _displacement(VanDerPolElement('osc', x=0.1, mu=0.1), until=200.0)
    assert np.abs(x[times > 150]).max() == pytest.approx(2.0, rel=1e-2)
    assert _period(times, x, 150.0) == pytest.approx(2 * np.pi * (1 + 0.1**2 / 16), rel=1e-3)


def test_van_der_pol_relaxes_with_its_asymptotic_period():
    """Test that a strongly damped oscillator relaxes with the period `(3 - 2 ln 2) mu + 7.014 mu ** (-1 / 3)`."""
    mu = 20.0
    times, x = _displacement(VanDerPolElement('osc', mu=mu), until=250.0, dt=0.005)
    expected = (3 - 2 * np.log(2)) * mu + 7.014 * mu ** (-1 / 3)
    assert _period(times, x, 50.0) == pytest.approx(expected, rel=2e-2)


def test_van_der_pol_starts_from_its_velocity():
    """Test that the oscillator starts from its initial velocity."""
    trajectory = (
        Simulation(System(elements=[VanDerPolElement('osc', x=0.0, x_d1=1.0)]), dt=0.001)
        .check()
        .prepare()
        .run(until=0.01)
        .trajectory_
    )
    assert trajectory.values('osc.x_d1')[0] == 1.0
    assert trajectory.values('osc.x')[-1] == pytest.approx(0.01, rel=1e-2)


def test_van_der_pol_locks_to_a_strong_forcing():
    """Test that a strong forcing near the natural frequency entrains the oscillator to the forcing period."""
    times, x = _displacement(VanDerPolElement('osc', mu=1.0, A=5.0, omega=1.2), until=300.0)
    assert _period(times, x, 150.0) == pytest.approx(2 * np.pi / 1.2, rel=1e-2)
