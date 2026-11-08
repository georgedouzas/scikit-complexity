"""Test the `_davidsen_ebel_bornholdt` module of `skcomplex.systems.mathematics.networks`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.systems.mathematics.networks import DavidsenEbelBornholdtSystem


def _graph(trajectory: Trajectory, frame: int) -> nx.Graph:
    """Return the network present at a frame as a NetworkX graph."""
    network = trajectory.network(frame)
    graph = nx.Graph()
    graph.add_nodes_from(network.nodes.tolist())
    graph.add_edges_from(zip(network.sources.tolist(), network.targets.tolist(), strict=True))
    return graph


def test_davidsen_ebel_bornholdt_settles_at_the_mean_degree_of_its_turnover_with_high_clustering():
    """Test that the steady mean degree is `1 + 1 / gamma` and that triadic closure keeps the clustering high."""
    trajectory = simulate(DavidsenEbelBornholdtSystem(200, turnover=0.2).prepare().system_, until=8000, random_state=0)
    graphs = [_graph(trajectory, frame) for frame in range(2000, 8001, 100)]
    degrees = [2 * graph.number_of_edges() / graph.number_of_nodes() for graph in graphs]
    assert {graph.number_of_nodes() for graph in graphs} == {200}
    assert np.mean(degrees) == pytest.approx(1 + 1 / 0.2, rel=0.15)
    assert np.mean([nx.average_clustering(graph) for graph in graphs]) > 0.2
