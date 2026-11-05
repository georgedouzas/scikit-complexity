"""Test the `_fractional` module of `skcomplex.systems.mathematics.fractional_dynamics`."""

import importlib

import numpy as np
import pytest

from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.types import FormKind
from skcomplex.systems import create_system
from skcomplex.systems.mathematics.fractional_dynamics import (
    fractional_lorenz,
    fractional_oscillator,
    fractional_relaxation,
    fractional_van_der_pol,
    relaxation_type,
)


def test_fractional_relaxation_is_slower_than_exponential():
    """Test that relaxation of order 0.5 stays above the exponential decay of order 1 at long times."""
    exponential = (
        Simulation(fractional_relaxation(order=1), dt=0.1).prepare().run(until=20.0).trajectory_.values('relaxation.x')
    )
    fractional = (
        Simulation(fractional_relaxation(order=0.5), dt=0.1)
        .prepare()
        .run(until=20.0)
        .trajectory_.values('relaxation.x')
    )
    assert np.all(fractional[50:] > exponential[50:])
    assert exponential[50] == pytest.approx(np.exp(-5.0), rel=1e-4)
    assert fractional[-1] > 0.1


def test_fractional_relaxation_order_one_is_ordinary():
    """Test that order 1 compiles to the ODE form and a fractional order to the fractional form."""
    assert compile_system(fractional_relaxation(order=1)).form == FormKind.ODE
    assert compile_system(fractional_relaxation(order=0.5)).form == FormKind.FRACTIONAL


def test_fractional_oscillator_damps_without_friction():
    """Test that the fractional oscillator of order 0.8 loses amplitude through its memory alone."""
    trajectory = (
        Simulation(fractional_oscillator(order=0.8), dt=0.1)
        .prepare()
        .run(until=30.0)
        .trajectory_.values('oscillator.x')
    )
    assert np.min(trajectory) < -0.1
    assert np.max(np.abs(trajectory[-50:])) < 0.2


def test_fractional_van_der_pol_keeps_oscillating():
    """Test that the fractional Van der Pol oscillator of order 0.95 keeps oscillating."""
    trajectory = (
        Simulation(fractional_van_der_pol(order=0.95), dt=0.1)
        .prepare()
        .run(until=40.0)
        .trajectory_.values('oscillator.x')
    )
    late = trajectory[-150:]
    assert np.max(late) > 1.0
    assert np.min(late) < -1.0


def test_fractional_lorenz_stays_on_bounded_attractor():
    """Test that the fractional Lorenz flow of order 0.995 stays bounded and keeps switching between both wings."""
    trajectory = Simulation(fractional_lorenz(), dt=0.05).prepare().run(until=40.0).trajectory_.values('lorenz.x')
    late = trajectory[-200:]
    assert np.all(np.abs(trajectory) < 30)
    assert np.sum(np.diff(np.sign(late)) != 0) >= 2


def test_element_types_of_orders_pickle_by_reference():
    """Test that the element types of every order are cached and pickle by reference."""
    assert relaxation_type(0.3) is relaxation_type(0.3)
    element_type = type(create_system('fractional_relaxation', order=0.37).elements[0])
    assert element_type is relaxation_type(0.37)
    assert getattr(importlib.import_module(element_type.__module__), element_type.__qualname__) is element_type
