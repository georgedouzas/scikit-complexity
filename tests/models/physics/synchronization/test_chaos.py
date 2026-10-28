"""Test the `_chaos` module of `skcomplex.models.physics.synchronization`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.physics.synchronization import (
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
    trajectory = Simulation(aggregate_averaging_synchronization(kappa=kappa), dt=1).check().run(until=100).trajectory_
    first, second = _series(trajectory, 'first.x', 'second.x')
    assert (np.abs(first - second)[-20:].max() < 1e-9) == synchronized
