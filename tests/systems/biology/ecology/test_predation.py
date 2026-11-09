"""Test the `_predation` module of `skcomplex.systems.biology.ecology`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.biology.ecology import (
    LotkaVolterraWithHarvestingSystem,
    NicholsonBaileySystem,
    RosenzweigMacarthurSystem,
    StochasticPredatorPreySystem,
)

from ..conftest import _run_until


def test_rosenzweig_macarthur_tips_only_at_fast_rates():
    """Test that the prey track the coexistence point under a slow rise of crowding and collapse under a fast one."""
    slow = _run_until(RosenzweigMacarthurSystem(rate=1e-4), 100, dt=0.5)
    fast = _run_until(RosenzweigMacarthurSystem(rate=0.004), 100, dt=0.5)
    assert slow.get_values('prey.x').min() > 4.5
    assert fast.get_values('prey.x').min() < 0.1


def test_rosenzweig_macarthur_prey_return_after_a_deep_collapse():
    """Test that the registered run follows the prey through a collapse far below any absolute tolerance and back."""
    trajectory = simulate(RosenzweigMacarthurSystem())
    reference = _run_until(
        RosenzweigMacarthurSystem(),
        RosenzweigMacarthurSystem.until,
        dt=RosenzweigMacarthurSystem.dt,
        solver=ScipyODESolver(method='DOP853', rtol=1e-12, atol=1e-300),
    )
    assert trajectory.get_values('prey.x').min() < 1e-20
    assert trajectory.get_values('prey.x')[-1] > 4.0
    np.testing.assert_allclose(trajectory.get_values('prey.x'), reference.get_values('prey.x'), rtol=1e-3, atol=1e-3)


def test_nicholson_bailey_equilibrium_is_unstable():
    """Test that the Nicholson-Bailey equilibrium is a fixed point that perturbations move away from."""
    k, a, c = 2.0, 0.05, 1.0
    hosts, parasitoids = k * np.log(k) / ((k - 1) * a * c), np.log(k) / a
    fixed = _run_until(NicholsonBaileySystem(hosts, parasitoids, k=k, a=a, c=c), 10, dt=1)
    assert fixed.get_values('hosts.x')[-1] == pytest.approx(hosts)
    perturbed = _run_until(NicholsonBaileySystem(hosts * 1.05, parasitoids, k=k, a=a, c=c), 20, dt=1)
    deviations = np.abs(perturbed.get_values('hosts.x') - hosts)
    assert deviations[10:].max() > 3 * deviations[0]


def test_harvesting_averages_match_shifted_equilibrium():
    """Test Volterra's principle: the cycle averages of harvested Lotka-Volterra are its shifted equilibrium."""
    trajectory = _run_until(LotkaVolterraWithHarvestingSystem(), 100, dt=0.05)
    assert trajectory.get_values('prey.x').mean() == pytest.approx((0.5 + 0.4) / 0.05, rel=0.03)
    assert trajectory.get_values('predators.x').mean() == pytest.approx((1.0 - 0.4) / 0.1, rel=0.03)


def test_stochastic_predator_prey_fluctuates_around_deterministic_equilibrium():
    """Test that demographic noise keeps integer populations that fluctuate around the deterministic equilibrium."""
    trajectory = _run_until(StochasticPredatorPreySystem(), 100, dt=1.0, random_state=0)
    prey = trajectory.get_values('prey.x')
    assert np.array_equal(prey, np.round(prey))
    assert prey[30:].mean() == pytest.approx(0.3 / (0.005 * (0.5 - 0.15)), rel=0.25)
