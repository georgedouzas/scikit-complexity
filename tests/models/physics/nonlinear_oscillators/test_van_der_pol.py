"""Test the `_van_der_pol` module of `skcomplex.models.physics.nonlinear_oscillators`."""

import numpy as np
import pytest

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.physics.nonlinear_oscillators import VanDerPolElement


def _run(**parameters):
    """Return the trajectory of one Van der Pol oscillator over a long run."""
    system = System(elements=[VanDerPolElement('oscillator', **parameters)])
    solver = ScipyODESolver(rtol=1e-9, atol=1e-11)
    return Simulation(system, dt=0.05, solver=solver).check().prepare().run(until=100.0).trajectory_


def test_unforced_oscillator_reaches_its_limit_cycle():
    """Test that a small displacement grows to the limit cycle of amplitude near 2."""
    trajectory = _run(x=0.1, mu=1.0)
    assert np.abs(trajectory.values('oscillator.x')[-400:]).max() == pytest.approx(2.0, abs=0.02)


def test_undamped_forced_oscillator_follows_the_forcing():
    """Test that without damping and from rest the forced oscillator follows `A (cos(w t) - cos(t)) / (1 - w^2)`."""
    trajectory = _run(x=0.0, mu=0.0, A=0.5, omega=2.0)
    expected = 0.5 * (np.cos(2.0 * trajectory.times) - np.cos(trajectory.times)) / (1 - 4.0)
    np.testing.assert_allclose(trajectory.values('oscillator.x'), expected, atol=1e-6)
