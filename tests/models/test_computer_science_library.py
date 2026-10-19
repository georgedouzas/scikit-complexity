"""Test the computer science models: automata, evolutionary computation, and neural networks."""

import dataclasses
import inspect
from math import comb

import numpy as np
import pytest

from skcomplex.core.base import ParameterError
from skcomplex.core.compile import compile_system
from skcomplex.core.runtime import Simulation
from skcomplex.models.computer_science.automata import (
    diffusion_automaton,
    elementary_automaton,
    game_of_life,
    highlife,
    noisy_automata,
    unbounded_alphabet_automaton,
)
from skcomplex.models.computer_science.evolutionary_computation import bit_string_evolution, robby_ga
from skcomplex.models.computer_science.neural_networks import hopfield

SIZE_NAMES = ('n_nodes', 'n_genes', 'size', 'n')
CONNECTIVITY_NAMES = ('k', 'K', 'connectivity', 'n_inputs')
N_NODES = 200


def _run(system, until, dt=1, random_state=0):
    """Return the trajectory of a system run with a frame duration and a random state."""
    return Simulation(system, dt=dt, random_state=random_state).run(until=until).trajectory_


def _damage(factory, connectivity, seed):
    """Return the share of nodes of a random Boolean network that a flip of one node changes after 50 steps."""
    parameters = inspect.signature(factory).parameters
    size = next((name for name in SIZE_NAMES if name in parameters), None)
    inputs = next((name for name in CONNECTIVITY_NAMES if name in parameters), None)
    if size is None or inputs is None:
        pytest.skip('The random Boolean network takes neither a known size nor a known connectivity argument.')
    options = {size: N_NODES, inputs: connectivity}
    if 'random_state' in parameters:
        options['random_state'] = seed
    problem = compile_system(factory(**options))
    name = problem.states.names[0]
    perturbed = problem.initial_state.copy()
    perturbed[problem.states.entry(name).start] = 1 - perturbed[problem.states.entry(name).start]
    finals = [
        np.asarray(_run(dataclasses.replace(problem, initial_state=initial), 50, random_state=seed)[name][-1])
        for initial in (problem.initial_state, perturbed)
    ]
    return float(np.mean(finals[0] != finals[1]))


def test_highlife_gives_birth_on_six_neighbours():
    """Test that HighLife, unlike Life, brings a dead cell with six live neighbours to life."""
    cells = np.zeros((7, 7))
    cells[2, 2:5] = cells[4, 2:5] = 1
    life = _run(game_of_life((7, 7), initial=cells), 1).lattice('cells.alive')[-1]
    high = _run(highlife((7, 7), initial=cells), 1).lattice('cells.alive')[-1]
    assert life[3, 3] == 0
    assert high[3, 3] == 1


def test_highlife_rejects_a_malformed_rule():
    """Test that a rule outside the notation `B<births>/S<survivals>` raises a parameter error."""
    with pytest.raises(ParameterError, match='B36/S23'):
        highlife((5, 5), rule='36/23')


def test_noisy_automata_follow_the_rule_without_noise_and_randomize_with_full_noise():
    """Test that the noisy automaton is the elementary one without noise and a coin flip at noise 1/2."""
    noiseless = _run(noisy_automata(18, noise=0.0), 20)['cells.alive']
    exact = _run(elementary_automaton(18, initial=noiseless[0]), 20)['cells.alive']
    np.testing.assert_array_equal(noiseless, exact)
    random = _run(noisy_automata(18, noise=0.5, size=201), 50)['cells.alive'][1:]
    assert random.mean() == pytest.approx(0.5, abs=0.02)
    assert 0 < random.mean(axis=1).std() < 0.1


def test_diffusion_automaton_conserves_and_spreads():
    """Test that discrete diffusion conserves the total content on the torus and flattens the square."""
    levels = _run(diffusion_automaton((30, 30), width=6), 50)['cells.level']
    np.testing.assert_allclose(levels.sum(axis=1), 36.0)
    assert levels[-1].max() < 0.5 * levels[0].max()


def test_unbounded_alphabet_automaton_builds_pascals_triangle():
    """Test that the additive automaton over the integers holds the binomial coefficients of its step."""
    values = _run(unbounded_alphabet_automaton(81), 40)['cells.value'][-1]
    np.testing.assert_array_equal(values[40:81], [comb(40, k) for k in range(41)])


def test_bit_string_evolution_climbs_the_landscape():
    """Test that selection raises the mean fitness of the population while its size stays constant."""
    trajectory = _run(bit_string_evolution(), 100)
    fitness = [trajectory.agents('genomes', frame).values['fitness'].mean() for frame in (0, 100)]
    assert fitness[1] > fitness[0] + 0.05
    assert trajectory['genomes.count'][-1] == 100


def test_robby_ga_learns_to_collect_cans():
    """Test that the genetic algorithm turns wall-bumping random strategies into ones that score near zero or above."""
    trajectory = _run(robby_ga(), 20)
    first, last = (trajectory.agents('strategies', frame).values['score'] for frame in (0, 20))
    assert first.mean() < -100
    assert last.mean() > first.mean() + 150
    assert last.max() > 0


def test_hopfield_recalls_a_stored_pattern_and_lowers_its_energy():
    """Test that the network recalls the first pattern from a corrupted copy, with an energy that never increases."""
    rng = np.random.default_rng(1)
    patterns = rng.choice([-1.0, 1.0], size=(3, 64))
    trajectory = _run(hopfield(patterns, noise=0.2, random_state=1), 5)
    snapshots = [trajectory.agents('neurons', frame) for frame in range(6)]
    weights = snapshots[0].values['weights']
    spins = [snapshot.values['spin'] for snapshot in snapshots]
    energies = [-spin @ weights @ spin / 2 for spin in spins]
    assert np.all(np.diff(energies) <= 1e-12)
    assert spins[0] @ patterns[0] < 64
    assert spins[-1] @ patterns[0] == 64


def test_random_boolean_network_phases():
    """Test that perturbations die out in Kauffman networks of connectivity 1 and spread at connectivity 3."""
    systems_biology = pytest.importorskip('skcomplex.models.biology.systems_biology')
    if not hasattr(systems_biology, 'random_boolean_network'):
        pytest.skip('The random Boolean network is not in the library yet.')
    factory = systems_biology.random_boolean_network
    ordered = np.mean([_damage(factory, 1, seed) for seed in range(5)])
    chaotic = np.mean([_damage(factory, 3, seed) for seed in range(5)])
    assert ordered < 0.05
    assert chaotic > 0.1
