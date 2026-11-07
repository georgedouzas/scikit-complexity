"""Test the `_node_merging` module of `skcomplex.systems.mathematics.networks`."""

import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import Simulation, Trajectory
from skcomplex.systems.mathematics.networks import node_merging


def _graph(trajectory: Trajectory, frame: int) -> nx.Graph:
    """Return the network present at a frame as a NetworkX graph."""
    network = trajectory.network(frame)
    graph = nx.Graph()
    graph.add_nodes_from(network.nodes.tolist())
    graph.add_edges_from(zip(network.sources.tolist(), network.targets.tolist(), strict=True))
    return graph


def test_node_merging_builds_hubs_at_a_fixed_size():
    """Test that merging keeps the size and lets the largest degree exceed ten times the mean and a random graph."""
    trajectory = Simulation(node_merging(500), random_state=0).prepare().run(until=10000).trajectory_
    graph = _graph(trajectory, -1)
    degrees = np.array([degree for _, degree in graph.degree()])
    random = nx.gnm_random_graph(500, graph.number_of_edges(), seed=0)
    assert graph.number_of_nodes() == 500
    assert degrees.max() > 10 * degrees.mean()
    assert degrees.max() > 3 * max(degree for _, degree in random.degree())
