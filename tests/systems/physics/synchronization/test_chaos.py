"""Test the `_chaos` module of `skcomplex.systems.physics.synchronization`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems import create_simulation
from skcomplex.systems.physics.synchronization import (
    aggregate_averaging_synchronization,
    coupled_rossler,
)

from ..conftest import _run, _series


@pytest.mark.parametrize(('k', 'synchronized'), [(0.5, True), (0.0, False)])
def test_identical_rossler_oscillators_synchronize_with_strong_coupling(k, synchronized):
    """Test that identical Rössler oscillators coupled strongly through their second variable synchronize completely."""
    trajectory = _run(coupled_rossler(k=k, mismatch=0.0), 80.0, 0.5, solver=ScipyODESolver(rtol=1e-8))
    first, second = _series(trajectory, 'oscillator_0.x1', 'oscillator_1.x1')
    assert (np.abs(first - second)[-20:].max() < 1e-4) == synchronized


@pytest.mark.parametrize(('kappa', 'synchronized'), [(0.45, True), (0.1, False)])
def test_aggregate_averaging_synchronizes_chaotic_maps(kappa, synchronized):
    """Test that logistic maps synchronize above the coupling `3 / 8` and stay apart well below it."""
    trajectory = Simulation(aggregate_averaging_synchronization(kappa=kappa), dt=1).prepare().run(until=100).trajectory_
    first, second = _series(trajectory, 'first.x', 'second.x')
    assert (np.abs(first - second)[-20:].max() < 1e-9) == synchronized


@pytest.mark.parametrize(
    ('kappa', 'delay', 'synchronized'),
    [(0.4, 0, True), (0.4, 1, True), (0.1, 1, False), (0.7, 1, False), (0.3, 2, True), (0.5, 2, False)],
)
def test_delay_coupled_bernoulli_maps_synchronize_inside_the_linear_window(kappa, delay, synchronized):
    """Test that the Bernoulli maps synchronize when every root of `z^(T+1) - r (1 - kappa) z^T + r kappa` is stable."""
    coefficients = np.zeros(delay + 2)
    coefficients[0], coefficients[1] = 1.0, -1.5 * (1 - kappa)
    coefficients[-1] += 1.5 * kappa
    assert (np.abs(np.roots(coefficients)).max() < 1) == synchronized
    system = aggregate_averaging_synchronization(kind='bernoulli', r=1.5, kappa=kappa, delay=delay)
    trajectory = Simulation(system, dt=1).prepare().run(until=400).trajectory_
    first, second = (np.ravel(trajectory.values(name)) for name in ('first.x', 'second.x'))
    difference = np.abs(first - second)
    assert (np.minimum(difference, 1 - difference)[-50:].max() < 1e-8) == synchronized
    assert first[-50:].std() > 0.1


def test_rossler_driven_lorenz_synchronizes_generally_above_the_critical_coupling():
    """Test that the two Lorenz copies driven by the Rössler system converge for `k = 8` and not for `k = 2`."""
    errors = []
    for k in (2.0, 8.0):
        trajectory = create_simulation('rossler_driven_lorenz', k=k).prepare().run().trajectory_
        squares = [
            (np.ravel(trajectory.values(f'response.{name}')) - np.ravel(trajectory.values(f'auxiliary.{name}'))) ** 2
            for name in 'xyz'
        ]
        errors.append(np.sqrt(sum(squares))[-2000:].mean())
    assert errors[0] > 5
    assert errors[1] < 1e-3
