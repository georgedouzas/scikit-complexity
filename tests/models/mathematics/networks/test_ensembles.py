"""Test the `_ensembles` module of `skcomplex.models.mathematics.networks`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import as_graph
from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.models.mathematics.networks import (
    network_attack,
    random_graph_ensembles,
)

from .conftest import _final_graph


def _giant(graph):
    """Return the share of the nodes of a graph in its largest component."""
    if not graph.number_of_nodes():
        return 0.0
    return max(len(component) for component in nx.connected_components(graph)) / graph.number_of_nodes()


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
