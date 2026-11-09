"""Test the `_evolutionary_learning` module of `skcomplex.systems.physics.criticality`."""

import numpy as np
import pytest

from skcomplex.core.analysis import events, fit_law
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.criticality import EvolutionarySKGameSystem

STEPS = 51_000


TRANSIENT = 3_000


SEED = 1


SWEEP = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


SWEEP_STEPS = 3_000


def test_evolutionary_sk_game_rules():
    """Test the symmetric couplings of variance 1/N, the uniform initial temperatures, and the fixed game."""
    agents = EvolutionarySKGameSystem(400, random_state=0).prepare().system_.elements[0]
    couplings = agents.values['couplings']
    assert np.array_equal(couplings, couplings.T)
    assert np.all(np.diag(couplings) == 0)
    assert couplings[np.triu_indices(400, 1)].var() * 400 == pytest.approx(1.0, rel=0.05)
    temperatures = np.asarray(agents.values['temperature'])
    assert temperatures.min() >= 0
    assert temperatures.max() <= 2
    fixed = EvolutionarySKGameSystem(10, temperature=0.3).prepare().system_.elements[0].values
    assert not fixed['evolving']
    assert np.all(np.asarray(fixed['temperature']) == 0.3)


def test_evolutionary_sk_game_replaces_the_least_fit():
    """Test that the least fit agent at a replacement step gets a new temperature and no other agent does."""
    trajectory = simulate(EvolutionarySKGameSystem(n_agents=32), until=10, random_state=0)
    temperatures, fitness = np.asarray(trajectory.get_values('agents.temperature')), np.asarray(
        trajectory.get_values('agents.fitness'),
    )
    for step in (3, 6, 9):
        changed = np.flatnonzero(temperatures[step + 1] != temperatures[step])
        assert changed.tolist() == [int(np.argmin(fitness[step]))]
    assert np.array_equal(temperatures[5], temperatures[4])


@pytest.mark.slow
def test_evolutionary_sk_game_self_organisation():
    """Test the evolved low temperatures, cascades of exponent 3/2, and a reward above every fixed temperature."""
    system = EvolutionarySKGameSystem(memory=0.75, random_state=SEED).prepare().system_
    evolved = simulate(system, until=STEPS, dt=1, random_state=SEED)
    temperatures = np.asarray(evolved.get_values('agents.temperature'))
    assert np.median(temperatures[-1]) < 0.5 < np.median(temperatures[0])
    fitness = np.asarray(evolved.get_values('agents.fitness'))[::3][TRANSIENT // 3 :]
    threshold = np.quantile(fitness.min(axis=1), 0.97)
    sizes = events((fitness < threshold).sum(axis=1), kind='excursions').sizes
    fit = fit_law(sizes, minimum=5, discrete=True)
    assert fit.exponent == pytest.approx(1.5, abs=0.15)
    reward = np.asarray(evolved.get_values('agents.reward'))[STEPS // 2 :].mean()
    for temperature in SWEEP:
        system = EvolutionarySKGameSystem(temperature=temperature, random_state=SEED).prepare().system_
        fixed = simulate(system, until=SWEEP_STEPS, dt=1, random_state=SEED)
        assert reward > np.asarray(fixed.get_values('agents.reward'))[SWEEP_STEPS // 2 :].mean(), temperature
