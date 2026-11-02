"""Test the `_predation` module of `skcomplex.models.biology.ecology`."""

import numpy as np
import pytest

from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models import describe, run
from skcomplex.models.biology.ecology import (
    lotka_volterra_with_harvesting,
    nicholson_bailey,
    rosenzweig_macarthur,
    stochastic_predator_prey,
)

from ..conftest import _run_until


def test_rosenzweig_macarthur_tips_only_at_fast_rates():
    """Test that the prey track the coexistence point under a slow rise of crowding and collapse under a fast one."""
    solver = describe('rosenzweig_macarthur').defaults['solver']
    slow = _run_until(rosenzweig_macarthur(rate=1e-4), 100, dt=0.5, solver=solver)
    fast = _run_until(rosenzweig_macarthur(rate=0.004), 100, dt=0.5, solver=solver)
    assert slow.values('prey.n').min() > 4.5
    assert fast.values('prey.n').min() < 0.1


def test_rosenzweig_macarthur_prey_return_after_a_deep_collapse():
    """Test that the registered run follows the prey through a collapse far below any absolute tolerance and back."""
    trajectory = run('rosenzweig_macarthur').trajectory_
    reference = _run_until(
        rosenzweig_macarthur(),
        describe('rosenzweig_macarthur').defaults['until'],
        dt=describe('rosenzweig_macarthur').defaults['dt'],
        solver=ScipyODESolver(method='DOP853', rtol=1e-12, atol=1e-300),
    )
    assert trajectory.values('prey.n').min() < 1e-20
    assert trajectory.values('prey.n')[-1] > 4.0
    np.testing.assert_allclose(trajectory.values('prey.n'), reference.values('prey.n'), rtol=1e-3, atol=1e-3)


def test_nicholson_bailey_equilibrium_is_unstable():
    """Test that the Nicholson-Bailey equilibrium is a fixed point that perturbations move away from."""
    k, a, c = 2.0, 0.05, 1.0
    hosts, parasitoids = k * np.log(k) / ((k - 1) * a * c), np.log(k) / a
    fixed = _run_until(nicholson_bailey(hosts, parasitoids, k=k, a=a, c=c), 10, dt=1)
    assert fixed.values('hosts.n')[-1] == pytest.approx(hosts)
    perturbed = _run_until(nicholson_bailey(hosts * 1.05, parasitoids, k=k, a=a, c=c), 20, dt=1)
    deviations = np.abs(perturbed.values('hosts.n') - hosts)
    assert deviations[10:].max() > 3 * deviations[0]


def test_harvesting_averages_match_shifted_equilibrium():
    """Test Volterra's principle: the cycle averages of harvested Lotka-Volterra are its shifted equilibrium."""
    trajectory = _run_until(lotka_volterra_with_harvesting(), 100, dt=0.05)
    assert trajectory.values('prey.n').mean() == pytest.approx((0.5 + 0.4) / 0.05, rel=0.03)
    assert trajectory.values('predators.n').mean() == pytest.approx((1.0 - 0.4) / 0.1, rel=0.03)


def test_stochastic_predator_prey_fluctuates_around_deterministic_equilibrium():
    """Test that demographic noise keeps integer populations that fluctuate around the deterministic equilibrium."""
    trajectory = _run_until(stochastic_predator_prey(), 100, dt=1.0, random_state=0)
    prey = trajectory.values('prey.n')
    assert np.array_equal(prey, np.round(prey))
    assert prey[30:].mean() == pytest.approx(0.3 / (0.005 * (0.5 - 0.15)), rel=0.25)
