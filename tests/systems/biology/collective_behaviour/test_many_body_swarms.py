"""Test the `_many_body_swarms` module of `skcomplex.systems.biology.collective_behaviour`."""

import itertools

import numpy as np
import pytest

from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import ManyBodyAlignmentSwarmSystem

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])


def _state_and_drift(backend, **arguments):
    """Return the positions, the velocities, and the accelerations of a swarm at its initial state."""
    system = ManyBodyAlignmentSwarmSystem(**arguments).prepare().system_
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    state = problem.initial_state
    drift = problem.drift(0.0, state, problem.initial_parameters)
    positions, velocities = problem.states.read(state, 'agents.x'), problem.states.read(state, 'agents.x_d1')
    return positions, velocities, problem.states.read(drift, 'agents.x_d1')


def _terms(positions, velocities, swarm):
    """Return the 1-body, 2-body, 3-body, and 4-body terms of Eq. 2 of every agent, evaluated directly.

    The offsets between agents are the minimum images in the periodic box of the swarm.
    """
    size = len(positions)
    side = np.asarray(swarm.prepare().system_.space.prepare().extent_)
    terms = np.zeros((4, *positions.shape))
    for i in range(size):
        v = velocities[i]
        terms[0, i] = (swarm.alpha - swarm.beta * v @ v) * v
        offsets = {j: positions[j] - positions[i] for j in range(size) if j != i}
        offsets = {j: offset - side * np.rint(offset / side) for j, offset in offsets.items()}
        around = [j for j, offset in offsets.items() if np.linalg.norm(offset) <= swarm.R]
        unit = {j: offsets[j] / np.linalg.norm(offsets[j]) for j in around}
        relative = {j: velocities[j] - v for j in around}
        cutoff = {j: (1 - np.linalg.norm(offsets[j]) ** 2 / swarm.R**2) ** 2 for j in around}
        weight = {j: cutoff[j] * np.exp(-np.linalg.norm(offsets[j]) / swarm.l_3) for j in around}
        for j in around:
            rho = np.linalg.norm(offsets[j])
            slope = -swarm.C_r / swarm.l_r * np.exp(-rho / swarm.l_r) + swarm.C_a / swarm.l_a * np.exp(-rho / swarm.l_a)
            terms[1, i] += cutoff[j] * (slope * unit[j] + swarm.gamma * np.exp(-rho / swarm.l_v) * relative[j])
        for j, k in itertools.combinations(around, 2):
            flank = (1 - unit[j] @ unit[k]) / 2 * (relative[j] + relative[k]) / 2
            terms[2, i] += swarm.kappa_v * weight[j] * weight[k] * flank
        for j, k, m in itertools.combinations(around, 3):
            spread = 1 - np.sum((unit[j] + unit[k] + unit[m]) ** 2) / 9
            mean = (relative[j] + relative[k] + relative[m]) / 3
            terms[3, i] += swarm.kappa_4 * weight[j] * weight[k] * weight[m] * spread * mean
    return terms


@BACKENDS
def test_pairwise_collective_has_the_one_and_two_body_terms(backend):
    """Test that without the 3-body and 4-body terms the drift is the sum of the 1-body and 2-body terms."""
    arguments = {'size': 10, 'kappa_v': 0.0, 'kappa_4': 0.0, 'random_state': 3}
    positions, velocities, drift = _state_and_drift(backend, **arguments)
    terms = _terms(positions, velocities, ManyBodyAlignmentSwarmSystem(**arguments))
    assert np.abs(terms[1]).sum() > 0
    np.testing.assert_allclose(drift, terms[0] + terms[1], rtol=0.0, atol=1e-12)


@BACKENDS
def test_body_order_terms_match_the_paper(backend):
    """Test that every body-order term of the drift equals a direct evaluation of Eq. 2."""
    arguments = {'size': 10, 'random_state': 3}
    positions, velocities, full = _state_and_drift(backend, kappa_v=2.0, kappa_4=4.0, **arguments)
    *_, pairwise = _state_and_drift(backend, kappa_v=0.0, kappa_4=0.0, **arguments)
    *_, flanking = _state_and_drift(backend, kappa_v=2.0, kappa_4=0.0, **arguments)
    *_, encircling = _state_and_drift(backend, kappa_v=0.0, kappa_4=4.0, **arguments)
    terms = _terms(positions, velocities, ManyBodyAlignmentSwarmSystem(kappa_v=2.0, kappa_4=4.0, **arguments))
    assert np.abs(terms[2]).sum() > 0
    assert np.abs(terms[3]).sum() > 0
    np.testing.assert_allclose(pairwise, terms[0] + terms[1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(flanking - pairwise, terms[2], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(encircling - pairwise, terms[3], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(full, terms.sum(axis=0), rtol=0.0, atol=1e-12)


@pytest.mark.slow
def test_swarm_moves_at_a_bounded_speed():
    """Test that the swarm with its defaults stays finite and near its preferred speed for 200 time units."""
    swarm = ManyBodyAlignmentSwarmSystem(40, random_state=0)
    trajectory = simulate(swarm, until=200.0, dt=1.0, backend=NumbaBackend())
    positions, velocities = trajectory.values('agents.x'), trajectory.values('agents.x_d1')
    speeds = np.linalg.norm(velocities, axis=-1) / np.sqrt(swarm.alpha / swarm.beta)
    assert np.isfinite(positions).all()
    assert np.isfinite(velocities).all()
    assert np.abs(speeds - 1).max() < 0.2
