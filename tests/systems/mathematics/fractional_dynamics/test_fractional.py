"""Test the `_fractional` module of `skcomplex.systems.mathematics.fractional_dynamics`."""

import numpy as np
import pytest

from skcomplex.core.engine.solvers import FractionalABMSolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.fractional_dynamics import (
    FractionalLorenzSystem,
    FractionalOscillatorSystem,
    FractionalRelaxationSystem,
    FractionalVanDerPolSystem,
)


def test_fractional_relaxation_is_slower_than_exponential():
    """Test that relaxation of order 0.5 stays above the exponential decay of order 1 at long times."""
    exponential = FractionalRelaxationSystem(order=1).prepare().simulate(20.0, dt=0.1).get_values('relaxation.x')
    fractional = FractionalRelaxationSystem(order=0.5).prepare().simulate(20.0, dt=0.1).get_values('relaxation.x')
    assert np.all(fractional[50:] > exponential[50:])
    assert exponential[50] == pytest.approx(np.exp(-5.0), rel=1e-4)
    assert fractional[-1] > 0.1


def test_fractional_relaxation_order_one_is_ordinary():
    """Test that order 1 prepares ordinary equations and a fractional order the Caputo solver."""
    ordinary = FractionalRelaxationSystem(order=1).prepare()
    fractional = FractionalRelaxationSystem(order=0.5).prepare()
    assert ordinary.form_ == 'ordinary differential equations'
    assert fractional.form_ == 'fractional differential equations'
    assert isinstance(fractional.solver_, FractionalABMSolver)


def test_fractional_relaxation_rejects_an_order_above_one():
    """Test that an order above 1 is rejected when the system prepares."""
    with pytest.raises(ValidationError, match=r'number in \(0, 1\]'):
        FractionalRelaxationSystem(order=1.5).prepare()


def test_fractional_oscillator_damps_without_friction():
    """Test that the fractional oscillator of order 0.8 loses amplitude through its memory alone."""
    trajectory = FractionalOscillatorSystem(order=0.8).prepare().simulate(30.0, dt=0.1).get_values('oscillator.x')
    assert np.min(trajectory) < -0.1
    assert np.max(np.abs(trajectory[-50:])) < 0.2


def test_fractional_van_der_pol_keeps_oscillating():
    """Test that the fractional Van der Pol oscillator of order 0.95 keeps oscillating."""
    system = FractionalVanDerPolSystem(order=0.95).prepare()
    late = system.simulate(40.0, dt=0.1).get_values('oscillator.x')[-150:]
    assert np.max(late) > 1.0
    assert np.min(late) < -1.0


def test_fractional_lorenz_stays_on_bounded_attractor():
    """Test that the fractional Lorenz flow of order 0.995 stays bounded and keeps switching between both wings."""
    trajectory = FractionalLorenzSystem().prepare().simulate(40.0, dt=0.05).get_values('lorenz.x')
    late = trajectory[-200:]
    assert np.all(np.abs(trajectory) < 30)
    assert np.sum(np.diff(np.sign(late)) != 0) >= 2


def test_set_params_changes_the_order():
    """Test that a new order reaches the run after the system prepares again."""
    system = FractionalRelaxationSystem(order=0.5).prepare()
    before = system.simulate(5.0, dt=1.0).get_values('relaxation.x')[-1]
    after = system.set_params(order=0.9).prepare().simulate(5.0, dt=1.0).get_values('relaxation.x')[-1]
    assert after < before


def test_oscillator_of_order_one_follows_the_cosine_without_damping():
    """Test that the undamped oscillator of order 1 follows `cos(omega t)`."""
    trajectory = FractionalOscillatorSystem(order=1.0, omega=2.0).prepare().simulate(5.0, dt=0.5)
    np.testing.assert_allclose(trajectory.get_values('oscillator.x'), np.cos(2.0 * trajectory.times), atol=1e-4)


def test_relaxation_of_order_one_decays_exponentially():
    """Test that relaxation of order 1 decays as `exp(-lam t)`."""
    trajectory = FractionalRelaxationSystem(x=2.0, order=1.0, lam=0.5).prepare().simulate(4.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('relaxation.x'), 2.0 * np.exp(-0.5 * trajectory.times), rtol=1e-4)


def test_van_der_pol_of_order_one_reaches_a_cycle_of_amplitude_two():
    """Test that the Van der Pol oscillator of order 1 started near rest grows to its cycle of amplitude near 2."""
    late = FractionalVanDerPolSystem(x=0.1, order=1.0).prepare().simulate(60.0, dt=0.1).get_values('oscillator.x')
    assert np.max(np.abs(late[-200:])) == pytest.approx(2.0, abs=0.1)
