"""Test the `_hypergraphs` module of `skcomplex.models.mathematics.networks`."""

import numpy as np
import pytest
import xgi

from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.models.mathematics.networks import hypergraph_random_walk


def _transitions(hypergraph: xgi.Hypergraph) -> np.ndarray:
    """Return the size-biased transition matrix of Carletti et al. from the incidence matrix of a hypergraph."""
    incidence = xgi.incidence_matrix(hypergraph, sparse=False)
    sizes = incidence.sum(axis=0)
    weights = incidence @ np.diag(sizes - 1) @ incidence.T
    np.fill_diagonal(weights, 0.0)
    return weights / weights.sum(axis=1, keepdims=True)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_hypergraph_random_walk_drift_follows_the_master_equation(backend):
    """Test that the drift equals `p T - p` for the size-biased transition matrix of the incidence matrix."""
    hypergraph = xgi.fast_random_hypergraph(12, [0.2, 0.05, 0.01], seed=2)
    problem = compile_system(hypergraph_random_walk(hypergraph), backend=backend)
    state = problem.initial_state.copy()
    state[:] = np.random.default_rng(0).dirichlet(np.ones(state.size))
    drift = problem.states.read(problem.drift(0.0, state, problem.initial_parameters), 'nodes.p')
    probabilities = problem.states.read(state, 'nodes.p')
    expected = probabilities @ _transitions(hypergraph) - probabilities
    assert np.allclose(drift, expected, atol=1e-12)


def test_hypergraph_random_walk_settles_on_the_hyperedge_weights():
    """Test that the probability keeps its sum and settles on the weights of the hyperedges of every node."""
    system = hypergraph_random_walk(seed=0)
    trajectory = Simulation(system, dt=1.0).check().prepare().run(until=60.0).trajectory_
    probabilities = np.asarray(trajectory.values('nodes.p'))
    weights = np.asarray(system.elements[0].values['D'])
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(probabilities[-1], weights / weights.sum(), atol=1e-6)
