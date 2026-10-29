"""Test the `_boolean` module of `skcomplex.models.biology.systems_biology`."""

import dataclasses
import inspect

import numpy as np
import pytest

from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.models.biology.systems_biology import (
    random_boolean_network,
    yeast_cell_cycle,
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


def _damage(k, seed, n_nodes=100, steps=40):
    """Return the Hamming distance after `steps` updates between a random Boolean network and a copy with one flip."""
    system = random_boolean_network(n_nodes, k, random_state=seed)
    problem = compile_system(system)
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


SIZE_NAMES = ('n_nodes', 'n_genes', 'size', 'n')


CONNECTIVITY_NAMES = ('k', 'K', 'connectivity', 'n_inputs')


N_NODES = 200


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
    problem = compile_system(factory(**options))
    name = problem.states.names[0]
    perturbed = problem.initial_state.copy()
    perturbed[problem.states.entry(name).start] = 1 - perturbed[problem.states.entry(name).start]
    finals = [
        np.asarray(_run(dataclasses.replace(problem, initial_state=initial), 50, random_state=seed).values(name)[-1])
        for initial in (problem.initial_state, perturbed)
    ]
    return float(np.mean(finals[0] != finals[1]))


def test_random_boolean_network_reaches_a_cycle():
    """Test that a random Boolean network returns to a state it visited, so it settles on an attractor."""
    states = Simulation(random_boolean_network(), dt=1).check().prepare().run(until=60).trajectory_.states
    seen = {tuple(row) for row in states[:30]}
    assert tuple(states[-1]) in seen or len({tuple(row) for row in states}) < len(states)


def test_random_boolean_network_takes_eight_inputs():
    """Test that every node reads eight distinct nodes through a table of 256 entries, and the network runs."""
    system = random_boolean_network(n_nodes=12, k=8)
    assert all(len(set(function.inputs)) == 8 and len(function.table) == 256 for function in system.interactions)
    assert len(Simulation(system, dt=1).check().prepare().run(until=50).trajectory_.times) == 51


def test_random_boolean_network_rejects_more_inputs_than_nodes():
    """Test that a node cannot read more inputs than the network has nodes."""
    with pytest.raises(ValidationError, match=r'k = 5 .*n_nodes = 4'):
        random_boolean_network(n_nodes=4, k=5)


def test_yeast_cell_cycle_matches_recorded_trajectory():
    """Test that the yeast network on one variadic role per node runs as the five-role version recorded before."""
    states = Simulation(yeast_cell_cycle(), dt=1).check().prepare().run(until=15).trajectory_.states
    assert [''.join(str(int(value)) for value in row) for row in states] == list(YEAST_RECORDED)


def test_yeast_cell_cycle_returns_to_g1():
    """Test that the yeast network passes through the M phase and ends in the stationary G1 state."""
    simulation = Simulation(yeast_cell_cycle(), dt=1).check().prepare().run(until=15)
    names = simulation.problem_.states.names
    active = [[name for name, value in zip(names, row, strict=True) if value] for row in simulation.trajectory_.states]
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
    systems_biology = pytest.importorskip('skcomplex.models.biology.systems_biology')
    if not hasattr(systems_biology, 'random_boolean_network'):
        pytest.skip('The random Boolean network is not in the library yet.')
    factory = systems_biology.random_boolean_network
    ordered = np.mean([_damage_share(factory, 1, seed) for seed in range(5)])
    chaotic = np.mean([_damage_share(factory, 3, seed) for seed in range(5)])
    assert ordered < 0.05
    assert chaotic > 0.1
