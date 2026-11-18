"""Test the `_kuramoto` module of `skcomplex.systems.physics.synchronization`."""

import math

import numpy as np
import pytest

from skcomplex.core.components.interactions import PairInteraction
from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.oscillators import PhaseOscillatorElement, SitedOscillatorElement
from skcomplex.systems.physics.synchronization import (
    DelayedKuramotoSystem,
    KuramotoCouplingInteraction,
    KuramotoSakaguchiChimeraSystem,
    KuramotoSystem,
    NonlocalRingChimeraSystem,
    OttAntonsenSystem,
)

from ..conftest import ACCURATE, _run, _series
from .conftest import _random_drift


def test_delayed_kuramoto_synchronizes():
    """Test that strongly coupled delayed oscillators lock their phases."""
    trajectory = DelayedKuramotoSystem(K=2.0, tau=1.0).prepare().simulate()
    order = np.abs(np.exp(1j * trajectory.states).mean(axis=1))
    assert order[0] < 0.1
    assert order[-1] > 0.99


@pytest.mark.parametrize(('K', 'synchronized'), [(4.0, True), (0.5, False)])
def test_kuramoto_synchronizes_above_the_critical_coupling(K, synchronized):
    """Test that the Kuramoto order parameter grows above the critical coupling and stays small below it."""
    trajectory = _run(
        KuramotoSystem(100, K=K),
        40.0,
        1.0,
        solver=ScipyODESolver(),
    )
    order = _series(trajectory, 'order')[0]
    assert (order[-10:].mean() > 0.8) == synchronized
    assert (order[-10:].mean() < 0.3) == (not synchronized)


def test_kuramoto_in_linear_time():
    """Test that Kuramoto has no space and no pair interaction, and its drift is the all-to-all sum."""
    system = KuramotoSystem(50)
    components = system.prepare().system_
    assert components.space is None
    assert not any(isinstance(interaction, PairInteraction) for interaction in components.interactions)
    problem, theta, rates = _random_drift(system)
    omega = problem.parameters.read(problem.initial_parameters, 'oscillators.omega')
    expected = omega + 4.0 / 50 * np.sin(theta[np.newaxis, :] - theta[:, np.newaxis]).sum(axis=1)
    np.testing.assert_allclose(rates, expected, atol=1e-12)


@pytest.mark.parametrize(('factor', 'synchronized'), [(0.5, False), (2.0, True)])
def test_kuramoto_transition(factor, synchronized):
    """Test that 2000 oscillators stay incoherent at half the critical coupling and synchronize at twice it."""
    critical = 2 * math.sqrt(2 / math.pi)
    system = KuramotoSystem(2000, K=factor * critical, solver=ScipyODESolver(method='RK45'))
    trajectory = system.prepare().simulate(until=100)
    order = _series(trajectory, 'order')[0][trajectory.times >= 50].mean()
    assert (order > 0.6) == synchronized
    assert (order < 0.1) == (not synchronized)


def test_kuramoto_sakaguchi_populations_form_a_chimera():
    """Test that every population reads both populations and that one synchronizes while the other drifts."""
    system = KuramotoSakaguchiChimeraSystem()
    assert {(item.target, item.source) for item in system.prepare().system_.interactions} == {
        (target, source) for target in ('first', 'second') for source in ('first', 'second')
    }
    trajectory = _run(system, 200.0, 5.0, solver=ScipyODESolver())
    first, second = _series(trajectory, 'first_order', 'second_order')
    coherent, incoherent = max(first[-1], second[-1]), min(first[-1], second[-1])
    assert coherent > 0.99
    assert 0.3 < incoherent < 0.95


def test_nonlocal_ring_rotates_rigidly_when_synchronized():
    """Test that a synchronized ring turns at `omega - (2R + 1) sin(alpha) / (2R)`, counting every neighbour once."""
    system = NonlocalRingChimeraSystem(30, R=4, omega=1.0).prepare().system_
    system.elements[0].theta = np.zeros(30)
    trajectory = system.set_params(solver=ACCURATE).prepare().simulate(until=2.0, dt=0.5)
    theta = _series(trajectory, 'ring.theta')[0]
    assert np.allclose(theta[-1], theta[-1][0])
    assert theta[-1][0] == pytest.approx(2.0 * (1.0 - 9 * math.sin(1.46) / 8), rel=1e-6)


@pytest.mark.parametrize(('K', 'expected'), [(2.0, math.sqrt(0.5)), (0.8, 0.0)])
def test_ott_antonsen_order_parameter_reaches_its_fixed_point(K, expected):
    """Test that the order parameter settles at `sqrt(1 - 2 Delta / K)` above `K_c = 2 Delta` and at 0 below."""
    trajectory = _run(OttAntonsenSystem(K=K), 80.0, 1.0)
    x, y = _series(trajectory, 'field.x', 'field.y')
    assert math.hypot(x[-1], y[-1]) == pytest.approx(expected, abs=1e-4)


THETA = np.array([0.1, 1.3, 2.9, -2.1, -0.8])


def _rate(element, coupling):
    """Return the rate of the phases of an element at its initial state."""
    problem = System(elements=[element], interactions=[coupling]).prepare()._problem
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    return problem.states.read(drift, f'{element.label}.theta')


@pytest.mark.parametrize('include_self', [False, True])
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_group_coupling_is_the_lagged_sine_sum(include_self, reduce):
    """Test that the group coupling, read from the group sums, is the sum of `K sin(theta_j - theta_i - alpha)`."""
    phases = PhaseOscillatorElement('phases', theta=THETA, omega=0.0, size=5)
    coupling = KuramotoCouplingInteraction(
        'coupling',
        target='phases',
        K=0.7,
        alpha=0.4,
        reduce=reduce,
        include_self=include_self,
    )
    terms = 0.7 * np.sin(THETA[np.newaxis, :] - THETA[:, np.newaxis] - 0.4)
    if not include_self:
        np.fill_diagonal(terms, 0.0)
    expected = terms.sum(axis=1) / (1 if reduce == 'sum' else 5 - (0 if include_self else 1))
    np.testing.assert_allclose(_rate(phases, coupling), expected, atol=1e-12)


def test_coupling_within_a_radius_on_a_periodic_ring():
    """Test that oscillators on a periodic ring couple to their neighbours within the radius and to themselves."""
    ring = SitedOscillatorElement(
        'ring',
        theta=THETA,
        omega=0.0,
        x=np.arange(5, dtype=float).reshape(-1, 1),
        size=5,
        space=ContinuousSpace((5.0,), boundary='periodic'),
    )
    coupling = KuramotoCouplingInteraction(
        'coupling',
        target='ring',
        K=0.7,
        alpha=0.4,
        include_self=True,
        neighborhood=RadiusNeighborhood(1.5),
    )
    expected = 0.7 * sum(np.sin(np.roll(THETA, shift) - THETA - 0.4) for shift in (-1, 0, 1))
    np.testing.assert_allclose(_rate(ring, coupling), expected, atol=1e-12)


def test_coupling_rejects_an_unknown_reduction():
    """Test that a reduction other than a sum or a mean is named in the error."""
    phases = PhaseOscillatorElement('phases', theta=THETA, size=5)
    coupling = KuramotoCouplingInteraction('coupling', target='phases', reduce='max')
    with pytest.raises(ValidationError, match="'max'"):
        System(elements=[phases], interactions=[coupling]).prepare()
