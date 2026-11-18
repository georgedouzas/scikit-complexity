"""Test the `_weighted_edge` module of `skcomplex.systems.biology.common.networks`."""

import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.systems.biology.common.networks import WeightedEdgeElement


def test_weighted_edge_holds_a_weight_and_a_mark():
    """Test that a weighted edge holds its weight and its mark, the mark set by default."""
    edge = WeightedEdgeElement('edge', w=0.5)
    assert (edge.w, edge.fresh) == (0.5, 1.0)


def test_weighted_edges_sit_on_the_edges_of_a_network():
    """Test that a group of weighted edges has one member per edge of its network."""
    space = NetworkSpace(nx.cycle_graph(4))
    edges = WeightedEdgeElement('edges', space=space, on='edges')
    assert (edges.space, edges.on) == (space, 'edges')
