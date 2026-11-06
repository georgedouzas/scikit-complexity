"""Test the `_hypergraphs` module of `skcomplex.systems.mathematics.networks`."""

import itertools

import numpy as np
import pytest
import xgi

from skcomplex.core.analysis import FitAnalysis
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system
from skcomplex.systems.mathematics.networks import growing_simplicial_complex, hypergraph_random_walk

from .conftest import _final_graph


def _transitions(hypergraph: xgi.Hypergraph) -> np.ndarray:
    """Return the size-biased transition matrix of Carletti and coworkers from the incidence matrix of a hypergraph."""
    incidence = xgi.incidence_matrix(hypergraph, sparse=False)
    sizes = incidence.sum(axis=0)
    weights = incidence @ np.diag(sizes - 1) @ incidence.T
    np.fill_diagonal(weights, 0.0)
    return weights / weights.sum(axis=1, keepdims=True)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_hypergraph_random_walk_drift_follows_the_master_equation(backend):
    """Test that the drift equals `p T - p` for the size-biased transition matrix of the incidence matrix."""
    hypergraph = xgi.fast_random_hypergraph(12, [0.2, 0.05, 0.01], seed=2)
    problem = PassCompiler(backend=backend).prepare(hypergraph_random_walk(hypergraph)).run().problem_
    state = problem.initial_state.copy()
    state[:] = np.random.default_rng(0).dirichlet(np.ones(state.size))
    drift = problem.states.read(problem.drift(0.0, state, problem.initial_parameters), 'nodes.p')
    probabilities = problem.states.read(state, 'nodes.p')
    expected = probabilities @ _transitions(hypergraph) - probabilities
    assert np.allclose(drift, expected, atol=1e-12)


def test_hypergraph_random_walk_settles_on_the_hyperedge_weights():
    """Test that the probability keeps its sum and settles on the weights of the hyperedges of every node."""
    system = hypergraph_random_walk(seed=0)
    trajectory = Simulation(system, dt=1.0).prepare().run(until=60.0).trajectory_
    probabilities = np.asarray(trajectory.values('nodes.p'))
    weights = np.asarray(system.elements[0].values['D'])
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(probabilities[-1], weights / weights.sum(), atol=1e-6)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_growing_simplicial_complex_attaches_one_triangle_per_step(backend):
    """Test that 50 steps from one triangle give 51 triangles along edges, each with its age in steps."""
    simulation = Simulation(growing_simplicial_complex(60), backend=backend, random_state=0).prepare()
    trajectory = simulation.run(until=50).trajectory_
    network = trajectory.network(-1)
    graph = network.graph()
    assert network.hyperedges.shape == (51, 3)
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (53, 3 + 2 * 50)
    assert set(range(3, 53)) <= set(network.hyperedges.ravel().tolist())
    assert all(graph.has_edge(*pair) for row in network.hyperedges for pair in itertools.combinations(row, 2))
    ages = np.asarray(trajectory.values('triangles.age')[-1])
    assert sorted(ages[~np.isnan(ages)].tolist()) == list(range(51))


def test_growing_simplicial_complex_degrees_follow_preferential_attachment():
    """Test that gluing triangles along uniform edges gives a degree tail exponent near 3."""
    graph, _ = _final_graph(growing_simplicial_complex(2003), 2000)
    assert graph.number_of_edges() == 3 + 2 * 2000
    degrees = np.array([degree for _, degree in graph.degree])
    fit = FitAnalysis(minimum=8, discrete=True, bootstrap=0).prepare(degrees).run()
    assert abs(fit.exponent_ - 3) < 0.4


def test_growing_simplicial_complex_is_created_by_name():
    """Test that the system is registered with variable groups on its nodes, edges, and triangles."""
    system = create_system('growing_simplicial_complex')
    assert [element.on for element in system.elements] == ['sites', 'edges', 'hyperedges']
    assert all(element.variable for element in system.elements)
