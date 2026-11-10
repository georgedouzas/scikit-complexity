"""Test the `_overload` module of `skcomplex.systems.mathematics.networks`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.networks import MotterLaiOverloadSystem


def _giant(seed, **arguments):
    """Return the share of the nodes in the largest component after the cascade."""
    trajectory = simulate(MotterLaiOverloadSystem(random_state=seed, **arguments), until=20)
    graph = trajectory.get_network(-1).get_graph()
    return max(len(component) for component in nx.connected_components(graph)) / 300


def test_an_attack_on_the_largest_load_collapses_a_scale_free_network_at_a_low_tolerance():
    """Test that a load attack at tolerance 0.1 leaves less than a third, and a random failure most of the network."""
    attacked = [_giant(seed, tolerance=0.1, attack='load') for seed in range(3)]
    failed = [_giant(seed, tolerance=0.1, attack='random') for seed in range(3)]
    assert np.mean(attacked) < 1 / 3
    assert np.mean(failed) > 0.9


def test_a_larger_tolerance_keeps_more_of_the_network():
    """Test that the largest component after a load attack grows with the tolerance."""
    shares = [np.mean([_giant(seed, tolerance=tolerance) for seed in range(3)]) for tolerance in (0.1, 0.5, 1.0)]
    assert shares[0] < shares[1] < shares[2]


def test_failed_nodes_carried_a_load_above_their_capacity():
    """Test that every node that fails after the attack carried more load on the network before than it holds."""
    system = MotterLaiOverloadSystem(tolerance=0.2, random_state=1).prepare().system_
    trajectory = simulate(system, until=6)
    capacity = system.elements[0].values['capacity']
    for frame in range(1, 5):
        before, after = trajectory.get_network(frame), trajectory.get_network(frame + 1)
        loads = nx.betweenness_centrality(before.get_graph(), normalized=False)
        failed = set(before.nodes.tolist()) - set(after.nodes.tolist())
        assert failed == {node for node in before.nodes.tolist() if loads[node] > capacity[node]}


def test_motter_lai_overload_rejects_an_unknown_attack():
    """Test that an unknown attack raises an error."""
    with pytest.raises(ValidationError, match='Unknown attack'):
        MotterLaiOverloadSystem(attack='nearest').prepare()
