"""Test the `_walks` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import math

import numpy as np
import pytest
from scipy import stats

from skcomplex.systems.mathematics.stochastic_processes import (
    AgeingRandomWalkSystem,
    PersistentRandomWalkSystem,
    RandomWalkSystem,
    RandomWalkWithMemorySystem,
    WeakeningOrnsteinUhlenbeckSystem,
)

from ..conftest import _final, _run


def _memory_msd(t, alpha, sigma):
    """Return the exact mean squared displacement of the random walk with memory after `t` steps from rest."""
    xx = xv = vv = 0.0
    for _ in range(t):
        xx, xv, vv = xx + 2 * xv + vv, alpha * (xv + vv), alpha**2 * vv + sigma**2
    return xx


def test_random_walk_spreads_diffusively_and_drifts_with_bias():
    """Test that the variance of unbiased walkers grows as `rate t` and biased walkers drift as `rate (2 bias - 1)`."""
    positions = _run(RandomWalkSystem(500), 50).values('walkers.x')[-1]
    assert abs(positions.mean()) < 1.0
    assert positions.var() == pytest.approx(50, rel=0.15)
    assert _run(RandomWalkSystem(500, bias=0.75), 50).values('walkers.x')[-1].mean() == pytest.approx(25, rel=0.1)


def test_weakening_ornstein_uhlenbeck_has_student_tails():
    """Test that the stationary density is Student's t with `2 nu - 1` degrees of freedom, `nu = k ℓ² / 2D`."""
    positions = _run(WeakeningOrnsteinUhlenbeckSystem(500, D=0.25), 20, dt=0.1).values('particles.x')[-100:]
    expected = stats.t(df=3).ppf(0.95) / math.sqrt(3)
    assert np.quantile(np.abs(positions), 0.9) == pytest.approx(expected, rel=0.1)


def test_ageing_random_walk_slows_its_up_steps():
    """Test that up steps grow linearly without ageing and as the square root of the steps for `nu = 1`."""
    ups = _run(AgeingRandomWalkSystem(200, nu=0.0), 100).get_agents('walkers', 100).values['ups']
    assert ups.mean() == pytest.approx(50, rel=0.05)
    ups = _run(AgeingRandomWalkSystem(200, nu=1.0), 100).get_agents('walkers', 100).values['ups']
    assert ups.mean() == pytest.approx(math.sqrt(100) - 1, rel=0.2)


def test_persistent_random_walk_diffuses_faster():
    """Test that persistent walkers have a mean squared displacement per step of `(1 + rho) / (1 - rho) = 4`."""
    positions = _final(PersistentRandomWalkSystem(n_walkers=2000, persistence=0.8), 1000).values('walkers.x')[-1]
    assert np.mean(positions**2) / 1000 == pytest.approx(4.0, rel=0.1)


def test_random_walk_with_memory_follows_the_exact_displacement():
    """Test that the mean squared displacement matches the exact second moments of the walk."""
    positions = _final(RandomWalkWithMemorySystem(n_walkers=4000), 500).values('walkers.x')[-1]
    assert np.mean(positions**2) == pytest.approx(_memory_msd(500, 0.5, 1.0), rel=0.1)
