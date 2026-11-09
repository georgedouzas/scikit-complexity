"""Test the `_fractional` module of `skcomplex.systems.mathematics.fractional_dynamics`."""

import importlib

import numpy as np
import pytest

from skcomplex.core.components.systems import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.types import FormKind
from skcomplex.systems.mathematics.fractional_dynamics import (
    FractionalLorenzSystem,
    FractionalOscillatorSystem,
    FractionalRelaxationSystem,
    FractionalVanDerPolSystem,
    OscillatorElement,
    RelaxationElement,
    VanDerPolElement,
    relaxation_type,
)


def test_fractional_relaxation_is_slower_than_exponential():
    """Test that relaxation of order 0.5 stays above the exponential decay of order 1 at long times."""
    exponential = simulate(FractionalRelaxationSystem(order=1), until=20.0, dt=0.1).values('relaxation.x')
    fractional = simulate(FractionalRelaxationSystem(order=0.5), until=20.0, dt=0.1).values('relaxation.x')
    assert np.all(fractional[50:] > exponential[50:])
    assert exponential[50] == pytest.approx(np.exp(-5.0), rel=1e-4)
    assert fractional[-1] > 0.1


def test_fractional_relaxation_order_one_is_ordinary():
    """Test that order 1 compiles to the ODE form and a fractional order to the fractional form."""
    assert (
        PassCompiler().prepare(FractionalRelaxationSystem(order=1).prepare().system_).run().problem_.form
        == FormKind.ODE
    )
    assert (
        PassCompiler().prepare(FractionalRelaxationSystem(order=0.5).prepare().system_).run().problem_.form
        == FormKind.FRACTIONAL
    )


def test_fractional_oscillator_damps_without_friction():
    """Test that the fractional oscillator of order 0.8 loses amplitude through its memory alone."""
    trajectory = simulate(FractionalOscillatorSystem(order=0.8), until=30.0, dt=0.1).values('oscillator.x')
    assert np.min(trajectory) < -0.1
    assert np.max(np.abs(trajectory[-50:])) < 0.2


def test_fractional_van_der_pol_keeps_oscillating():
    """Test that the fractional Van der Pol oscillator of order 0.95 keeps oscillating."""
    trajectory = simulate(FractionalVanDerPolSystem(order=0.95), until=40.0, dt=0.1).values('oscillator.x')
    late = trajectory[-150:]
    assert np.max(late) > 1.0
    assert np.min(late) < -1.0


def test_fractional_lorenz_stays_on_bounded_attractor():
    """Test that the fractional Lorenz flow of order 0.995 stays bounded and keeps switching between both wings."""
    trajectory = simulate(FractionalLorenzSystem(), until=40.0, dt=0.05).values('lorenz.x')
    late = trajectory[-200:]
    assert np.all(np.abs(trajectory) < 30)
    assert np.sum(np.diff(np.sign(late)) != 0) >= 2


def test_element_types_of_orders_pickle_by_reference():
    """Test that the element types of every order are cached and pickle by reference."""
    assert relaxation_type(0.3) is relaxation_type(0.3)
    element_type = type(FractionalRelaxationSystem(order=0.37).prepare().system_.elements[0])
    assert element_type is relaxation_type(0.37)
    assert getattr(importlib.import_module(element_type.__module__), element_type.__qualname__) is element_type


def test_oscillator_element_follows_the_cosine_without_damping():
    """Test that the undamped oscillator of order 1 follows `cos(omega t)`."""
    system = System(elements=[OscillatorElement('oscillator', omega=2.0)])
    trajectory = simulate(system, until=5.0, dt=0.5)
    np.testing.assert_allclose(trajectory.values('oscillator.x'), np.cos(2.0 * trajectory.times), atol=1e-4)


def test_relaxation_element_decays_exponentially():
    """Test that relaxation of order 1 decays as `exp(-lam t)`."""
    system = System(elements=[RelaxationElement('relaxation', x=2.0, lam=0.5)])
    trajectory = simulate(system, until=4.0, dt=1.0)
    np.testing.assert_allclose(trajectory.values('relaxation.x'), 2.0 * np.exp(-0.5 * trajectory.times), rtol=1e-4)


def test_van_der_pol_element_reaches_a_cycle_of_amplitude_two():
    """Test that the Van der Pol oscillator of order 1 started near rest grows to its cycle of amplitude near 2."""
    system = System(elements=[VanDerPolElement('oscillator', x=0.1, mu=1.0)])
    trajectory = simulate(system, until=60.0, dt=0.1)
    late = trajectory.values('oscillator.x')[-200:]
    assert np.max(np.abs(late)) == pytest.approx(2.0, abs=0.1)
