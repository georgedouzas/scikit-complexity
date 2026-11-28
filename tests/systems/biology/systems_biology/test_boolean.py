"""Test the `_boolean` module of `skcomplex.systems.biology.systems_biology`."""

import dataclasses
import inspect
import time

import numpy as np
import pytest

from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.systems_biology import (
    RandomBooleanNetworkSystem,
    RandomBooleanNodeElement,
    TruthTablesInteraction,
    YeastCellCycleSystem,
)

from ...conftest import _run

YEAST_RECORDED = (
    '10001000100',
    '01101000100',
    '01111000100',
    '01110000000',
    '01110001000',
    '01110001011',
    '00010011011',
    '00000110011',
    '00000110111',
    '00000110101',
    '00001110100',
    '00001100100',
    *('00001000100',) * 4,
)
SIZE_NAMES = ('n_nodes', 'n_genes', 'size', 'n')
CONNECTIVITY_NAMES = ('k', 'K', 'connectivity', 'n_inputs')
N_NODES = 200


def _damage(k, seed, n_nodes=100, steps=40):
    """Return the Hamming distance after `steps` updates between a random Boolean network and a copy with one flip."""
    system = RandomBooleanNetworkSystem(n_nodes, k, random_state=seed).prepare().system_
    problem = PassCompiler().prepare(system).run().problem_
    start = problem.initial_state.copy()
    flipped = start.copy()
    flipped[0] = 1 - flipped[0]
    finals = []
    for initial in (start, flipped):
        state = initial
        for step in range(steps):
            state = problem.update(float(step), state, problem.initial_parameters)
        finals.append(state)
    return float(np.abs(finals[0] - finals[1]).sum())


def _damage_share(factory, connectivity, seed):
    """Return the share of nodes of a random Boolean network that a flip of one node changes after 50 steps."""
    parameters = inspect.signature(factory).parameters
    size = next((name for name in SIZE_NAMES if name in parameters), None)
    inputs = next((name for name in CONNECTIVITY_NAMES if name in parameters), None)
    if size is None or inputs is None:
        pytest.skip('The random Boolean network takes neither a known size nor a known connectivity argument.')
    options = {size: N_NODES, inputs: connectivity}
    if 'random_state' in parameters:
        options['random_state'] = seed
    problem = PassCompiler().prepare(factory(**options)).run().problem_
    name = problem.states.names[0]
    perturbed = problem.initial_state.copy()
    perturbed[problem.states.get_entry(name).start] = 1 - perturbed[problem.states.get_entry(name).start]
    finals = [
        np.asarray(
            _run(dataclasses.replace(problem, initial_state=initial), 50, random_state=seed).get_values(name)[-1],
        )
        for initial in (problem.initial_state, perturbed)
    ]
    return float(np.mean(finals[0] != finals[1]))


def test_random_boolean_network_reaches_a_cycle():
    """Test that a random Boolean network returns to a state it visited, so it settles on an attractor."""
    states = RandomBooleanNetworkSystem().prepare().simulate(dt=1, until=60).states
    seen = {tuple(row) for row in states[:30]}
    assert tuple(states[-1]) in seen or len({tuple(row) for row in states}) < len(states)


def test_random_boolean_network_takes_eight_inputs():
    """Test that every node reads eight distinct nodes through a table of 256 entries, and the network runs."""
    system = RandomBooleanNetworkSystem(n_nodes=12, k=8).prepare().system_
    nodes = system.elements[0]
    assert all(len(set(row)) == 8 for row in nodes.inputs.tolist())
    assert nodes.table.shape == (12, 256)
    assert len(system.prepare().simulate(dt=1, until=50).times) == 51


def test_random_boolean_network_runs_a_large_network_in_seconds():
    """Test that a network of 10_000 nodes with four inputs each runs 50 steps in seconds on one state of nodes."""
    start = time.perf_counter()
    values = RandomBooleanNetworkSystem(10_000, 4, random_state=3).prepare().simulate().get_values('nodes.x')
    assert time.perf_counter() - start < 30
    assert values.shape == (51, 10_000)


def test_random_boolean_network_follows_its_tables():
    """Test that every synchronous step sets each node to the entry of its table at the pattern of its inputs."""
    system = RandomBooleanNetworkSystem(50, 3, random_state=1).prepare().system_
    nodes = system.elements[0]
    values = system.prepare().simulate(until=5, dt=1).get_values('nodes.x').astype(int)
    codes = values[:-1][:, nodes.inputs] @ np.array([4, 2, 1])
    np.testing.assert_array_equal(values[1:], nodes.table[np.arange(50), codes])


def test_asynchronous_update_sets_one_random_node_at_a_time():
    """Test that n single updates of negating nodes flip the nodes drawn an odd number of times, near 43 percent."""
    size = 20_000
    nodes = RandomBooleanNodeElement(
        'nodes',
        x=0,
        inputs=np.arange(size)[:, None],
        table=np.tile([1, 0], (size, 1)),
        size=size,
    )
    interaction = TruthTablesInteraction('functions', target='nodes', k=1, update='asynchronous')
    system = System('negations', elements=[nodes], interactions=[interaction], backend=NumpyBackend())
    values = system.prepare().simulate(until=1, dt=1, random_state=0).get_values('nodes.x')
    assert np.mean(values[-1]) == pytest.approx((1 - np.exp(-2)) / 2, abs=0.01)


def test_annealed_network_follows_the_annealed_approximation():
    """Test that two copies of an annealed network share its draws and their distance follows Derrida and Pomeau."""
    size, steps = 20_000, 8
    system = RandomBooleanNetworkSystem(size, 2, annealed=True, random_state=0).prepare().system_
    start = np.asarray(system.elements[0].x)
    flipped = start.copy()
    flipped[: size // 5] = 1 - flipped[: size // 5]
    runs = [
        system.set_params(nodes__x=initial).prepare().simulate(until=steps, dt=1, random_state=4).get_values('nodes.x')
        for initial in (start, flipped)
    ]
    distances = np.mean(runs[0] != runs[1], axis=1)
    expected = [0.2]
    for _ in range(steps):
        expected.append(0.5 * (1 - (1 - expected[-1]) ** 2))
    np.testing.assert_allclose(distances, expected, atol=0.015)


def test_truth_tables_reject_an_unknown_update():
    """Test that the update of the truth tables is synchronous or asynchronous."""
    with pytest.raises(ValidationError, match=r"Unknown update 'random'"):
        RandomBooleanNetworkSystem(update='random').prepare()


def test_random_boolean_network_rejects_more_inputs_than_nodes():
    """Test that a node cannot read more inputs than the network has nodes."""
    with pytest.raises(ValidationError, match=r'k = 5 .*n_nodes = 4'):
        RandomBooleanNetworkSystem(n_nodes=4, k=5).prepare()


def test_yeast_cell_cycle_matches_recorded_trajectory():
    """Test that the yeast network on one variadic role per node runs as the five-role version recorded before."""
    states = YeastCellCycleSystem().prepare().simulate(dt=1, until=15).states
    assert [''.join(str(int(value)) for value in row) for row in states] == list(YEAST_RECORDED)


def test_yeast_cell_cycle_returns_to_g1():
    """Test that the yeast network passes through the M phase and ends in the stationary G1 state."""
    trajectory = YeastCellCycleSystem().prepare().system_.simulate(until=15, dt=1)
    names = trajectory.problem.states.names
    active = [[name for name, value in zip(names, row, strict=True) if value] for row in trajectory.states]
    assert any('Clb1_2.x' in names for names in active)
    assert active[-1] == active[-2] == ['Cdh1.x', 'Sic1.x']


def test_random_boolean_network_damage_follows_its_phase():
    """Test that one flip dies out at in-degree 1, stays near constant at 2, and spreads at 3."""
    damage = {k: np.mean([_damage(k, seed) for seed in range(12)]) for k in (1, 2, 3)}
    assert damage[1] < 1.0
    assert damage[2] < 5.0
    assert damage[3] > 8.0
    assert damage[1] < damage[2] < damage[3]


def test_random_boolean_network_phases():
    """Test that perturbations die out in Kauffman networks of connectivity 1 and spread at connectivity 3."""
    systems_biology = pytest.importorskip('skcomplex.systems.biology.systems_biology')
    if not hasattr(systems_biology, 'random_boolean_network'):
        pytest.skip('The random Boolean network is not in the library yet.')
    factory = systems_biology.random_boolean_network
    ordered = np.mean([_damage_share(factory, 1, seed) for seed in range(5)])
    chaotic = np.mean([_damage_share(factory, 3, seed) for seed in range(5)])
    assert ordered < 0.05
    assert chaotic > 0.1
