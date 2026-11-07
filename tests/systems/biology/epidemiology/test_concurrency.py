"""Test the `_concurrency` module of `skcomplex.systems.biology.epidemiology`."""

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.biology.epidemiology import pair_formation_sis, watts_may_concurrency

from ..conftest import _run_precise


def test_watts_may_conserves_people_and_spreads_faster_with_longer_partnerships():
    """Test that the compartments keep their total and that longer partnerships raise the early infections."""
    short = _run_precise(watts_may_concurrency(tau=0.5), 10, dt=0.5)
    long = _run_precise(watts_may_concurrency(tau=3.0), 10, dt=0.5)
    total = sum(long.values(f'population.{state}') for state in 'SEIR')
    np.testing.assert_allclose(total, 1.0, atol=1e-8)
    assert long.values('population.I')[-1] > 1.5 * short.values('population.I')[-1]


def test_watts_may_exposed_wait_the_exposed_period():
    """Test that nobody turns infectious before the exposed period of the first infections ends."""
    trajectory = _run_precise(watts_may_concurrency(T=2.0, nu=0.0), 2.0, dt=0.25)
    np.testing.assert_allclose(trajectory.values('population.I'), 0.01, rtol=1e-6)


def test_pair_formation_concurrency_raises_the_prevalence():
    """Test that concurrent partners at the same contact rate keep more people infected than serial monogamy."""

    def prevalence(k: int) -> float:
        runs = []
        for seed in range(2):
            simulation = Simulation(pair_formation_sis(n=300, k=k, seed=seed), dt=0.5, random_state=seed)
            runs.append(simulation.prepare().run(until=25).trajectory_.values('people.i')[20:].mean())
        return float(np.mean(runs))

    assert prevalence(4) > prevalence(1) + 0.03
