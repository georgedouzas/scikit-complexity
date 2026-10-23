"""Test the `_kuramoto` module of `skcomplex.models.physics.synchronization`."""

import math

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models import describe
from skcomplex.models.physics.synchronization import (
    delayed_kuramoto,
    kuramoto,
    kuramoto_sakaguchi_chimera,
    nonlocal_ring_chimera,
    ott_antonsen,
)

from ..conftest import _run, _series
from .conftest import _random_drift


def test_delayed_kuramoto_synchronizes():
    """Test that strongly coupled delayed oscillators lock their phases."""
    trajectory = Simulation(delayed_kuramoto(K=2.0, tau=1.0), dt=0.1).run(until=50).trajectory_
    order = np.abs(np.exp(1j * trajectory.states).mean(axis=1))
    assert order[0] < 0.1
    assert order[-1] > 0.99


@pytest.mark.parametrize(('K', 'synchronized'), [(4.0, True), (0.5, False)])
def test_kuramoto_synchronizes_above_the_critical_coupling(K, synchronized):
    """Test that the Kuramoto order parameter grows above the critical coupling and stays small below it."""
    trajectory = _run(
        built := kuramoto(100, K=K),
        40.0,
        1.0,
        solver=ScipyODE(),
        observables=describe('kuramoto').observables_of(built),
    )
    order = _series(trajectory, 'order')[0]
    assert (order[-10:].mean() > 0.8) == synchronized
    assert (order[-10:].mean() < 0.3) == (not synchronized)


def test_kuramoto_in_linear_time():
    """Test that Kuramoto has no space and no pair interaction, and its drift is the all-to-all sum."""
    system = kuramoto(50)
    assert system.space is None
    assert not any(type(interaction).over == 'pairs' for interaction in system.interactions)
    problem, theta, rates = _random_drift(system)
    omega = problem.parameters.read(problem.initial_parameters, 'oscillators.omega')
    expected = omega + 4.0 / 50 * np.sin(theta[np.newaxis, :] - theta[:, np.newaxis]).sum(axis=1)
    np.testing.assert_allclose(rates, expected, atol=1e-12)


@pytest.mark.parametrize(('factor', 'synchronized'), [(0.5, False), (2.0, True)])
def test_kuramoto_transition(factor, synchronized):
    """Test that 2000 oscillators stay incoherent at half the critical coupling and synchronize at twice it."""
    critical = 2 * math.sqrt(2 / math.pi)
    trajectory = (
        Simulation(
            built := kuramoto(2000, K=factor * critical),
            dt=0.5,
            observables=describe('kuramoto').observables_of(built),
        )
        .run(until=100)
        .trajectory_
    )
    order = _series(trajectory, 'order')[0][trajectory.times >= 50].mean()
    assert (order > 0.6) == synchronized
    assert (order < 0.1) == (not synchronized)


def test_kuramoto_sakaguchi_populations_form_a_chimera():
    """Test that every population reads both populations and that one synchronizes while the other drifts."""
    system = kuramoto_sakaguchi_chimera()
    assert {(item.target, item.source) for item in system.interactions} == {
        (target, source) for target in ('first', 'second') for source in ('first', 'second')
    }
    trajectory = _run(
        system,
        200.0,
        5.0,
        solver=ScipyODE(),
        observables=describe('kuramoto_sakaguchi_chimera').observables_of(system),
    )
    first, second = _series(trajectory, 'first_order', 'second_order')
    coherent, incoherent = max(first[-1], second[-1]), min(first[-1], second[-1])
    assert coherent > 0.99
    assert 0.3 < incoherent < 0.95


def test_nonlocal_ring_rotates_rigidly_when_synchronized():
    """Test that a synchronized ring turns at `omega - (2R + 1) sin(alpha) / (2R)`, counting every neighbour once."""
    system = nonlocal_ring_chimera(30, R=4, omega=1.0)
    system.elements[0].values['theta'] = np.zeros(30)
    trajectory = _run(system, 2.0, 0.5, observables=describe('nonlocal_ring_chimera').observables_of(system))
    theta = _series(trajectory, 'ring.theta')[0]
    assert np.allclose(theta[-1], theta[-1][0])
    assert theta[-1][0] == pytest.approx(2.0 * (1.0 - 9 * math.sin(1.46) / 8), rel=1e-6)


@pytest.mark.parametrize(('K', 'expected'), [(2.0, math.sqrt(0.5)), (0.8, 0.0)])
def test_ott_antonsen_order_parameter_reaches_its_fixed_point(K, expected):
    """Test that the order parameter settles at `sqrt(1 - 2 Delta / K)` above `K_c = 2 Delta` and at 0 below."""
    trajectory = _run(ott_antonsen(K=K), 80.0, 1.0)
    x, y = _series(trajectory, 'field.x', 'field.y')
    assert math.hypot(x[-1], y[-1]) == pytest.approx(expected, abs=1e-4)
