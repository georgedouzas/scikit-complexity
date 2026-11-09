"""Test the `_ensembles` module of `skcomplex.systems.mathematics.networks`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.networks import (
    ConfigurationModelSystem,
    InterdependentNetworkCascadeSystem,
    KroneckerGraphSystem,
    NetworkAttackSystem,
    RandomGraphEnsemblesSystem,
)

from ..conftest import _run
from .conftest import _final_graph


def _giant(graph):
    """Return the share of the nodes of a graph in its largest component."""
    if not graph.number_of_nodes():
        return 0.0
    return max(len(component) for component in nx.connected_components(graph)) / graph.number_of_nodes()


def test_targeted_attacks_break_the_giant_component_faster_than_failures():
    """Test that removing hubs destroys the giant component while as many random failures leave it mostly whole."""
    trajectory = simulate(NetworkAttackSystem(500, strategy='targeted'), until=24, dt=1, random_state=0)
    frame = next(frame for frame in range(25) if len(trajectory.get_network(frame).nodes) <= 425)
    attacked = trajectory.get_network(frame).graph()
    removed = 500 - attacked.number_of_nodes()
    failures = NetworkAttackSystem(500, strategy='random', rate=removed / 500 / frame)
    failed, _ = _final_graph(failures, frame)
    assert _giant(attacked) < 0.5
    assert _giant(failed) > 0.9
    assert len(trajectory.get_network(0).nodes) == 500


def test_network_attack_rejects_an_unknown_strategy():
    """Test that an unknown strategy raises an error that lists the strategies."""
    with pytest.raises(ValidationError, match='targeted'):
        NetworkAttackSystem(strategy='flood').prepare()


def test_erdos_renyi_construction_gives_a_giant_component_past_mean_degree_one():
    """Test that the edge by edge Erdős-Rényi construction reaches its mean degree and a giant component after 1."""
    system = RandomGraphEnsemblesSystem(400, ensemble='erdos_renyi', mean_degree=4, steps=40)
    trajectory = simulate(system, until=40, dt=1.0, random_state=1)
    means = np.array([2 * len(trajectory.get_network(frame).sources) / 400 for frame in range(41)])
    giants = np.array([_giant(trajectory.get_network(frame).graph()) for frame in range(41)])
    assert means[-1] == pytest.approx(4, rel=0.1)
    assert giants[np.argmin(np.abs(means - 0.5))] < 0.1
    assert giants[-1] > 0.9


def test_watts_strogatz_construction_shortens_paths_and_keeps_clustering():
    """Test that rewiring a fifth of a ring shortens its paths far more than it lowers its clustering."""
    ring = nx.watts_strogatz_graph(300, 6, 0.0)
    rewired, _ = _final_graph(RandomGraphEnsemblesSystem(300, ensemble='watts_strogatz', mean_degree=6, steps=50), 50)
    assert rewired.number_of_edges() == ring.number_of_edges()
    paths = nx.average_shortest_path_length(rewired) / nx.average_shortest_path_length(ring)
    clustering = nx.average_clustering(rewired) / nx.average_clustering(ring)
    assert paths < 0.3
    assert clustering > 0.4


def test_random_graph_ensembles_reject_an_unknown_ensemble():
    """Test that an unknown ensemble raises an error that lists the ensembles."""
    with pytest.raises(ValidationError, match='watts_strogatz'):
        RandomGraphEnsemblesSystem(ensemble='lattice').prepare()


def test_kronecker_graph_multiplies_the_degrees():
    """Test that the degree of the node `(i, j)` of a Kronecker graph is the product of the degrees of `i` and `j`."""
    first, second = nx.barabasi_albert_graph(12, 2, seed=1), nx.cycle_graph(5)
    graph = KroneckerGraphSystem(first, second).prepare().system_.space.graph
    assert graph.number_of_nodes() == 60
    for node, pair in graph.nodes(data='pair'):
        assert graph.degree(node) == first.degree(pair[0]) * second.degree(pair[1])
    degrees = [degree for _, degree in graph.degree]
    assert np.mean(degrees) == pytest.approx(np.mean([d for _, d in first.degree]) * 2)


def test_configuration_model_keeps_the_degree_sequence_in_a_simple_graph():
    """Test that rewiring removes every self-loop and multilink while every node keeps its degree."""
    degrees = [6, 5, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1]
    graph = ConfigurationModelSystem(degrees, random_state=3).prepare().system_.space.graph
    assert sorted(dict(graph.degree).values(), reverse=True) == degrees
    assert nx.number_of_selfloops(graph) == 0


def test_configuration_model_shows_the_friendship_paradox():
    """Test that the end of a random link has the mean degree `<k²> / <k>`, above the mean degree `<k>`."""
    graph = ConfigurationModelSystem(n_nodes=3000).prepare().system_.space.graph
    degrees = np.array([degree for _, degree in graph.degree])
    ends = np.array([graph.degree(end) for edge in graph.edges for end in edge])
    assert ends.mean() == pytest.approx(np.mean(degrees**2) / degrees.mean(), rel=1e-9)
    assert ends.mean() > 1.3 * degrees.mean()


def test_interdependent_networks_keep_a_mutual_giant_component_above_the_threshold_only():
    """Test that the cascade stops at `S = p (1 - exp(-k S))²` above `p_c = 2.4554 / k` and collapses below it."""
    survived = _run(InterdependentNetworkCascadeSystem(2000, p=0.85), 40).get_values('nodes.active')
    share = 0.85
    for _ in range(500):
        share = 0.85 * (1 - np.exp(-4 * share)) ** 2
    assert survived[-1].mean() == pytest.approx(share, abs=0.04)
    collapsed = _run(InterdependentNetworkCascadeSystem(2000, p=0.55), 40).get_values('nodes.active')
    assert collapsed[-1].sum() == 0
    assert collapsed[1].mean() > 0.4
