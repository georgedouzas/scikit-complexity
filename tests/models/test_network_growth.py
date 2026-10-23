"""Test the growing networks, the random graph ensembles, and the attacks on networks."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import as_graph, fit_power_law
from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.models import create
from skcomplex.models.mathematics.networks import (
    barabasi_albert,
    holme_kim,
    network_attack,
    random_graph_ensembles,
    vertex_copying,
)


def _final_graph(system, until, seed=0):
    """Return the graph of the last frame of a run of a network whose topology changes."""
    trajectory = Simulation(system, dt=max(1, until // 10), random_state=seed).run(until=until).trajectory_
    return as_graph(trajectory.network(-1)), trajectory


def _giant(graph):
    """Return the share of the nodes of a graph in its largest component."""
    if not graph.number_of_nodes():
        return 0.0
    return max(len(component) for component in nx.connected_components(graph)) / graph.number_of_nodes()


def test_barabasi_albert_grows_a_scale_free_network():
    """Test that Barabási-Albert growth adds one node with two edges per step and a degree tail exponent near 3."""
    graph, trajectory = _final_graph(barabasi_albert(3003), 3000)
    assert graph.number_of_nodes() == 3003
    assert graph.number_of_edges() == 3 + 2 * 3000
    degrees = np.array([degree for _, degree in graph.degree])
    fit = fit_power_law(degrees, xmin=8, discrete=True, bootstrap=0)
    assert abs(fit.alpha - 3) < 0.4
    assert np.nanmax(trajectory['nodes.degree'][-1]) == pytest.approx(degrees.max(), abs=2)


def test_holme_kim_keeps_clustering_that_barabasi_albert_loses():
    """Test that triad formation keeps the clustering high where preferential attachment alone lets it vanish."""
    clustered, _ = _final_graph(holme_kim(1004, triad=0.9), 1000)
    attached, _ = _final_graph(barabasi_albert(1003, edges=3, seed_nodes=3), 1000)
    assert clustered.number_of_edges() == pytest.approx(attached.number_of_edges(), rel=0.02)
    assert nx.average_clustering(clustered) > 0.3
    assert nx.average_clustering(attached) < 0.1


def test_vertex_copying_makes_hubs_that_random_attachment_lacks():
    """Test that copying edges gives a much larger largest degree than joining random nodes."""
    copied, _ = _final_graph(vertex_copying(1004, randomness=0.1), 1000)
    uniform, _ = _final_graph(vertex_copying(1004, randomness=1.0), 1000)
    largest = max(degree for _, degree in copied.degree)
    assert largest > 2 * max(degree for _, degree in uniform.degree)


def test_targeted_attacks_break_the_giant_component_faster_than_failures():
    """Test that removing hubs destroys the giant component while as many random failures leave it mostly whole."""
    trajectory = Simulation(network_attack(500, strategy='targeted'), dt=1, random_state=0).run(until=24).trajectory_
    frame = next(frame for frame in range(25) if len(trajectory.network(frame).nodes) <= 425)
    attacked = as_graph(trajectory.network(frame))
    removed = 500 - attacked.number_of_nodes()
    failures = network_attack(500, strategy='random', rate=removed / 500 / frame)
    failed, _ = _final_graph(failures, frame)
    assert _giant(attacked) < 0.5
    assert _giant(failed) > 0.9
    assert len(trajectory.network(0).nodes) == 500


def test_network_attack_rejects_an_unknown_strategy():
    """Test that an unknown strategy raises an error that lists the strategies."""
    with pytest.raises(ValidationError, match='targeted'):
        network_attack(strategy='flood')


def test_erdos_renyi_construction_gives_a_giant_component_past_mean_degree_one():
    """Test that the edge by edge Erdős-Rényi construction reaches its mean degree and a giant component after 1."""
    system = random_graph_ensembles(400, ensemble='erdos_renyi', mean_degree=4, steps=40)
    trajectory = Simulation(system, dt=1.0, random_state=1).run(until=40).trajectory_
    means = np.array([2 * len(trajectory.network(frame).sources) / 400 for frame in range(41)])
    giants = np.array([_giant(as_graph(trajectory.network(frame))) for frame in range(41)])
    assert means[-1] == pytest.approx(4, rel=0.1)
    assert giants[np.argmin(np.abs(means - 0.5))] < 0.1
    assert giants[-1] > 0.9


def test_watts_strogatz_construction_shortens_paths_and_keeps_clustering():
    """Test that rewiring a fifth of a ring shortens its paths far more than it lowers its clustering."""
    ring = nx.watts_strogatz_graph(300, 6, 0.0)
    rewired, _ = _final_graph(random_graph_ensembles(300, ensemble='watts_strogatz', mean_degree=6, steps=50), 50)
    assert rewired.number_of_edges() == ring.number_of_edges()
    paths = nx.average_shortest_path_length(rewired) / nx.average_shortest_path_length(ring)
    clustering = nx.average_clustering(rewired) / nx.average_clustering(ring)
    assert paths < 0.3
    assert clustering > 0.4


def test_random_graph_ensembles_reject_an_unknown_ensemble():
    """Test that an unknown ensemble raises an error that lists the ensembles."""
    with pytest.raises(ValidationError, match='watts_strogatz'):
        random_graph_ensembles(ensemble='lattice')


@pytest.mark.parametrize('name', ['barabasi_albert', 'holme_kim', 'vertex_copying'])
def test_growth_models_are_created_by_name(name):
    """Test that the growth models are registered and start with their seed network."""
    system = create(name)
    assert system.components[0].variable
    assert system.components[1].on == 'edges'
