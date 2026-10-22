"""Test the fractional dynamics models of the library."""

import pickle  # nosec B403: the tests pickle and load only objects they create

import numpy as np
import pytest

from skcomplex.core.base import Form
from skcomplex.core.compile import compile_system
from skcomplex.core.runtime import Simulation
from skcomplex.models import create
from skcomplex.models.mathematics.fractional_dynamics import (
    fractional_lorenz,
    fractional_oscillator,
    fractional_relaxation,
    fractional_van_der_pol,
    relaxation_type,
)


def test_fractional_relaxation_is_slower_than_exponential():
    """Test that relaxation of order 0.5 stays above the exponential decay of order 1 at long times."""
    exponential = Simulation(fractional_relaxation(order=1), dt=0.1).run(until=20.0).trajectory_['relaxation.x']
    fractional = Simulation(fractional_relaxation(order=0.5), dt=0.1).run(until=20.0).trajectory_['relaxation.x']
    assert np.all(fractional[50:] > exponential[50:])
    assert exponential[50] == pytest.approx(np.exp(-5.0), rel=1e-4)
    assert fractional[-1] > 0.1


def test_fractional_relaxation_order_one_is_ordinary():
    """Test that order 1 compiles to the ODE form and a fractional order to the fractional form."""
    assert compile_system(fractional_relaxation(order=1)).form is Form.ODE
    assert compile_system(fractional_relaxation(order=0.5)).form is Form.FRACTIONAL


def test_fractional_oscillator_damps_without_friction():
    """Test that the fractional oscillator of order 0.8 loses amplitude through its memory alone."""
    trajectory = Simulation(fractional_oscillator(order=0.8), dt=0.1).run(until=30.0).trajectory_['oscillator.x']
    assert np.min(trajectory) < -0.1
    assert np.max(np.abs(trajectory[-50:])) < 0.2


def test_fractional_van_der_pol_keeps_oscillating():
    """Test that the fractional Van der Pol oscillator of order 0.95 keeps oscillating."""
    trajectory = Simulation(fractional_van_der_pol(order=0.95), dt=0.1).run(until=40.0).trajectory_['oscillator.x']
    late = trajectory[-150:]
    assert np.max(late) > 1.0
    assert np.min(late) < -1.0


def test_fractional_lorenz_stays_on_bounded_attractor():
    """Test that the fractional Lorenz flow of order 0.99 stays bounded and switches between both wings."""
    trajectory = Simulation(fractional_lorenz(), dt=0.05).run(until=30.0).trajectory_['lorenz.x']
    assert np.all(np.abs(trajectory) < 30)
    assert np.min(trajectory) < -5
    assert np.max(trajectory) > 5


def test_element_types_of_orders_pickle_by_reference():
    """Test that the element types of every order are cached and pickle by reference."""
    assert relaxation_type(0.3) is relaxation_type(0.3)
    system = create('fractional_relaxation', order=0.37)
    loaded = pickle.loads(pickle.dumps(system))  # noqa: S301  # nosec B301: loads data the test just dumped
    assert type(loaded.elements[0]) is relaxation_type(0.37)
