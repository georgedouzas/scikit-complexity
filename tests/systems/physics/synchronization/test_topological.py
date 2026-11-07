"""Test the `_topological` module of `skcomplex.systems.physics.synchronization`."""

import numpy as np

from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import describe
from skcomplex.systems.physics.synchronization import topological_kuramoto

SYNCHRONIZATION_GAP = 0.3


def _final_order(coupling):
    """Return the mean over the last frames of the order parameter of the link phases of a run at a coupling."""
    system = topological_kuramoto(coupling=coupling)
    observables = describe('topological_kuramoto').observables_of(system)
    simulation = Simulation(system, dt=0.2, observables=observables).prepare().run(until=40.0)
    return float(np.mean(simulation.trajectory_.values('link_order')[-20:]))


def test_link_synchronization_grows_with_coupling():
    """Test that the order of the link phases well above the transition exceeds that at zero coupling by 0.3."""
    assert _final_order(8.0) - _final_order(0.0) >= SYNCHRONIZATION_GAP


def test_rates_follow_the_incidence_of_the_complex():
    """Test the rates of the phases against the boundary matrices of Eq.

    6 of the review.
    """
    system = topological_kuramoto(8, density=0.8, coupling=1.5)
    problem = PassCompiler().prepare(system).run().problem_
    state = problem.initial_state
    rates = problem.drift(0.0, state, problem.initial_parameters)
    space = system.elements[0].space
    edges, faces = space.incidences_[:2]
    theta, phi = state[:8], state[8:]
    parameters = problem.parameters
    omega_nodes = parameters.read(problem.initial_parameters, 'nodes.omega')
    omega_links = parameters.read(problem.initial_parameters, 'links.omega')
    down_order = np.abs(np.exp(-1j * (edges.T @ phi)).mean())
    node_order = np.abs(np.exp(1j * theta).mean())
    expected_nodes = omega_nodes - 1.5 * down_order * (edges.T @ np.sin(edges @ theta))
    expected_links = omega_links - 1.5 * (node_order * (edges @ np.sin(edges.T @ phi)) + faces.T @ np.sin(faces @ phi))
    assert np.allclose(rates[:8], np.ravel(expected_nodes))
    assert np.allclose(rates[8:], np.ravel(expected_links))


def test_nodes_links_triangles_rates_follow_eq_8_of_the_review():
    """Test the rates of the Nodes-Links-Triangles system against the boundary matrices and order parameters of Eq.

    8.
    """
    system = topological_kuramoto(8, density=0.8, coupling=1.5, model='NLT')
    problem = PassCompiler().prepare(system).run().problem_
    state = problem.initial_state
    rates = problem.drift(0.0, state, problem.initial_parameters)
    edges, faces = system.elements[0].space.incidences_[:2]
    theta, phi = state[:8], state[8:]
    omega_links = problem.parameters.read(problem.initial_parameters, 'links.omega')
    down_order = np.abs(np.exp(1j * (edges.T @ phi)).mean())
    up_order = np.abs(np.exp(1j * (faces @ phi)).mean())
    node_order = np.abs(np.exp(1j * theta).mean())
    expected = omega_links - 1.5 * (
        node_order * up_order * (edges @ np.sin(edges.T @ phi)) + down_order * (faces.T @ np.sin(faces @ phi))
    )
    assert faces.shape[0] > 0
    assert np.allclose(rates[8:], np.ravel(expected))


def test_nodes_links_triangles_system_stays_incoherent_where_the_nodes_links_system_synchronizes():
    """Test that at a weak coupling the nodes synchronize in the Nodes-Links system but not with the triangles."""
    orders = {}
    for model in ('NL', 'NLT'):
        system = topological_kuramoto(coupling=0.5, model=model)
        observables = describe('topological_kuramoto').observables_of(system)
        simulation = Simulation(system, dt=0.2, observables=observables).prepare().run(until=40.0)
        orders[model] = float(np.mean(simulation.trajectory_.values('node_order')[-20:]))
    assert orders['NL'] > 0.9
    assert orders['NLT'] < 0.4
