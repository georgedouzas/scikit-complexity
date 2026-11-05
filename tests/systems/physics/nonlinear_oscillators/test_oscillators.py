"""Test the `_oscillators` module of `skcomplex.systems.physics.nonlinear_oscillators`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems import create_simulation
from skcomplex.systems.physics.nonlinear_oscillators import (
    adler_equation,
    forced_van_der_pol,
    stuart_landau,
)

from ..conftest import _run, _series, _upward_crossings


def test_van_der_pol_reaches_limit_cycle():
    """Test that the Van der Pol oscillator settles on a cycle of amplitude near 2."""
    x = create_simulation('van_der_pol').prepare().run().trajectory_.values('oscillator.x')
    assert 1.9 < np.abs(x[-200:]).max() < 2.1


def test_henon_heiles_conserves_energy():
    """Test that the Hénon-Heiles star keeps its energy within its invariant tolerance."""
    trajectory = create_simulation('henon_heiles').prepare().run().trajectory_
    x, y, px, py = (trajectory.values(name) for name in ('star.x', 'star.y', 'star.px', 'star.py'))
    energy = (px**2 + py**2 + x**2 + y**2) / 2 + x**2 * y - y**3 / 3
    assert np.abs(energy - energy[0]).max() < 1e-6
    np.testing.assert_allclose(trajectory.values('star.energy'), energy)


def test_henon_heiles_stays_bounded_below_the_escape_energy():
    """Test that the star started at the energy 1/12 stays bounded to time 1000 within its energy tolerance."""
    y, py = 0.0, 0.0
    px = math.sqrt(2 / 12)
    trajectory = create_simulation('henon_heiles', x=0.0, y=y, px=px, py=py).prepare().run(until=1000.0).trajectory_
    x, y = (np.asarray(trajectory.values(name)) for name in ('star.x', 'star.y'))
    energy = np.asarray(trajectory.values('star.energy'))
    assert energy[0] == pytest.approx(1 / 12)
    assert np.hypot(x, y).max() < 1.0
    assert np.ptp(energy) < 1e-6


def test_stuart_landau_settles_on_its_limit_cycle():
    """Test that the Stuart-Landau oscillator settles on the circle of radius `sqrt(mu)`."""
    trajectory = _run(stuart_landau(mu=0.5, beta=0.5), 40.0, 0.1)
    x, y = _series(trajectory, 'oscillator.x', 'oscillator.y')
    phase = np.unwrap(np.arctan2(y, x))
    assert np.hypot(x, y)[-1] == pytest.approx(math.sqrt(0.5), rel=1e-6)
    assert np.diff(phase[-20:]).mean() / 0.1 == pytest.approx(1.0 - 0.25, rel=1e-4)


def test_forced_van_der_pol_locks_to_a_strong_force():
    """Test that a strong force entrains the Van der Pol oscillator to its own frequency."""
    trajectory = _run(forced_van_der_pol(A=1.0, omega=1.1), 120.0, 0.05, solver=ScipyODESolver(rtol=1e-8))
    crossings = _upward_crossings(trajectory.times, _series(trajectory, 'oscillator.x')[0])
    assert np.diff(crossings[-5:]).mean() == pytest.approx(2 * math.pi / 1.1, rel=1e-3)


@pytest.mark.parametrize('detuning', [0.5, 1.5])
def test_adler_equation_locks_inside_the_tongue_and_slips_outside(detuning):
    """Test that the phase difference locks for `|Δω| < ε` and slips with period `2π / sqrt(Δω² - ε²)` beyond."""
    trajectory = _run(adler_equation(detuning=detuning), 60.0, 0.05)
    phi = _series(trajectory, 'difference.phi')[0]
    if detuning < 1:
        assert math.sin(phi[-1]) == pytest.approx(-detuning, abs=1e-6)
        return
    slips = np.interp(2 * math.pi * np.arange(1, 8), phi - phi[0], trajectory.times)
    assert np.diff(slips).mean() == pytest.approx(2 * math.pi / math.sqrt(detuning**2 - 1), rel=1e-3)
