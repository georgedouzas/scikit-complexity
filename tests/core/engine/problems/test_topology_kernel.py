"""Test the `topology_kernel` module."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.engine.problems._topology_kernel import TopologyKernel
from skcomplex.core.systems import System


def test_topology_kernel_get_independent():
    """Test `TopologyKernel._get_independent` on a small case."""
    weights, allowed = np.array([1.0, 0.0, 2.0, 1.0]), np.array([True, True, True, False])
    assert TopologyKernel._get_independent(weights, allowed, np.random.default_rng(0)) == [0, 2]


def test_topology_kernel_changes_edges():
    """Test `TopologyKernel._changes_edges` on a small case."""
    rewire, removal = TopologyRequestSpec('rewire'), TopologyRequestSpec('remove-node')
    assert (TopologyKernel._changes_edges(rewire), TopologyKernel._changes_edges(removal)) == (True, False)


def _grown_graph(closing: float, until: int = 30) -> nx.Graph:
    """Return the last graph of a complete seed on four nodes that grows by new nodes with further edges."""
    space = NetworkSpace(nx.complete_graph(4))
    nodes = Element('nodes', values={'age': 0.0}, space=space)
    links = Element('links', space=space, on='edges')
    request = TopologyRequestSpec('add-node', further=2, closing=closing)
    growth = Interaction('growth', laws=[UpdateLaw(request, 1)])
    system = System('system', elements=[nodes, links], interactions=[growth], space=space)
    return system.prepare().simulate(until=until, random_state=0).get_network(-1).get_graph()


@pytest.mark.parametrize('closing', [0.0, 1.0])
def test_topology_kernel_adds_further_edges_in_the_step_a_node_joins(closing):
    """Test that a new node holds its first and further edges in the frame after it joins."""
    graph = _grown_graph(closing)
    assert graph.number_of_nodes() == 34
    assert all(graph.degree(node) >= 3 for node in range(4, 34))


def test_topology_kernel_closes_triangles_with_the_neighbours_of_the_first_contact():
    """Test that further edges that always close triangles join neighbours of the first contact."""
    graph = _grown_graph(1.0)
    for node in range(4, 34):
        joined = [other for other in graph.neighbors(node) if other < node]
        assert any(all(graph.has_edge(first, other) for other in joined if other != first) for first in joined)
