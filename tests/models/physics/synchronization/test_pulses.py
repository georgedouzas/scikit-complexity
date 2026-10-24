"""Test the `_pulses` module of `skcomplex.models.physics.synchronization`."""

import numpy as np
import pytest

from skcomplex.engine.solvers import ScipyODE
from skcomplex.models import describe
from skcomplex.models.physics.synchronization import (
    mirollo_strogatz,
    winfree,
)

from ..conftest import _run, _series
from .conftest import _random_drift


def test_winfree_mean_influence():
    """Test that the Winfree drift is the response to the mean influence, with no pair or self-influence term."""
    system = winfree(50)
    assert len(system.interactions) == 1
    assert type(system.interactions[0]).over != 'pairs'
    problem, theta, rates = _random_drift(system)
    omega = problem.parameters.read(problem.initial_parameters, 'oscillators.omega')
    np.testing.assert_allclose(rates, omega - np.sin(theta) * 0.5 / 50 * (1 + np.cos(theta)).sum(), atol=1e-12)


@pytest.mark.parametrize(('K', 'dead'), [(2.0, True), (0.1, False)])
def test_winfree_oscillators_die_under_strong_coupling(K, dead):
    """Test that strong Winfree coupling stops the oscillators and weak coupling lets them turn."""
    trajectory = _run(
        built := winfree(20, K=K),
        60.0,
        1.0,
        solver=ScipyODE(),
        observables=describe('winfree').observables_of(built),
    )
    theta = _series(trajectory, 'oscillators.theta')[0]
    speed = np.abs(theta[-1] - theta[-2]).mean()
    assert (speed < 1e-3) == dead
    assert (speed > 0.5) == (not dead)


def test_mirollo_strogatz_oscillators_fire_in_unison():
    """Test that the integrate-and-fire oscillators, members of one group, end up firing in unison."""
    system = mirollo_strogatz(6)
    assert system.elements[0].size == 6
    trajectory = _run(system, 30.0, 0.1, solver=ScipyODE())
    charges = _series(trajectory, 'oscillators.x')[0]
    assert np.ptp(charges[-1]) < 1e-9
    assert np.ptp(charges[0]) > 0.1
    assert {event.label for event in trajectory.events} == {'fire'}
