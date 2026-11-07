"""Test the `_evolutionary_learning` module of `skcomplex.systems.physics.criticality`."""

import numpy as np
import pytest

from skcomplex.core.analysis import EventAnalysis, fit_law
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system
from skcomplex.systems.physics.criticality import evolutionary_sk_game

STEPS = 51_000


TRANSIENT = 3_000


SEED = 1


SWEEP = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


SWEEP_STEPS = 3_000


def test_evolutionary_sk_game_rules():
    """Test the symmetric couplings of variance 1/N, the uniform initial temperatures, and the fixed game."""
    agents = evolutionary_sk_game(400, random_state=0).elements[0]
    couplings = agents.values['couplings']
    assert np.array_equal(couplings, couplings.T)
    assert np.all(np.diag(couplings) == 0)
    assert couplings[np.triu_indices(400, 1)].var() * 400 == pytest.approx(1.0, rel=0.05)
    temperatures = np.asarray(agents.values['temperature'])
    assert temperatures.min() >= 0
    assert temperatures.max() <= 2
    fixed = evolutionary_sk_game(10, temperature=0.3).elements[0].values
    assert not fixed['evolving']
    assert np.all(np.asarray(fixed['temperature']) == 0.3)


def test_evolutionary_sk_game_replaces_the_least_fit():
    """Test that the least fit agent at a replacement step gets a new temperature and no other agent does."""
    trajectory = (
        Simulation(create_system('evolutionary_sk_game', n_agents=32), dt=1, random_state=0)
        .prepare()
        .run(until=10)
        .trajectory_
    )
    temperatures, fitness = np.asarray(trajectory.values('agents.temperature')), np.asarray(
        trajectory.values('agents.fitness'),
    )
    for step in (3, 6, 9):
        changed = np.flatnonzero(temperatures[step + 1] != temperatures[step])
        assert changed.tolist() == [int(np.argmin(fitness[step]))]
    assert np.array_equal(temperatures[5], temperatures[4])


@pytest.mark.slow
def test_evolutionary_sk_game_self_organisation():
    """Test the evolved low temperatures, cascades of exponent 3/2, and a reward above every fixed temperature."""
    system = create_system('evolutionary_sk_game', memory=0.75, random_state=SEED)
    evolved = Simulation(system, dt=1, random_state=SEED).prepare().run(until=STEPS).trajectory_
    temperatures = np.asarray(evolved.values('agents.temperature'))
    assert np.median(temperatures[-1]) < 0.5 < np.median(temperatures[0])
    fitness = np.asarray(evolved.values('agents.fitness'))[::3][TRANSIENT // 3 :]
    threshold = np.quantile(fitness.min(axis=1), 0.97)
    sizes = EventAnalysis('excursions').prepare((fitness < threshold).sum(axis=1)).run().sizes_
    fit = fit_law(sizes, minimum=5, discrete=True)
    assert fit.exponent == pytest.approx(1.5, abs=0.15)
    reward = np.asarray(evolved.values('agents.reward'))[STEPS // 2 :].mean()
    for temperature in SWEEP:
        system = create_system('evolutionary_sk_game', temperature=temperature, random_state=SEED)
        fixed = Simulation(system, dt=1, random_state=SEED).prepare().run(until=SWEEP_STEPS).trajectory_
        assert reward > np.asarray(fixed.values('agents.reward'))[SWEEP_STEPS // 2 :].mean(), temperature
