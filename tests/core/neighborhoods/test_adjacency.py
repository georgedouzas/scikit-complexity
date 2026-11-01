"""Test the `_adjacency` module of `skcomplex.core.neighborhoods`."""

import networkx as nx

from skcomplex.core.neighborhoods import AdjacencyNeighborhood
from skcomplex.core.spaces import NetworkSpace


def test_adjacency_neighbourhood_puts_the_node_first_when_it_is_included():
    """Test that an adjacency neighbourhood with its centre reads the node before its neighbours."""
    space = NetworkSpace(nx.star_graph(3)).check().prepare()
    assert space._get_neighbors(2, AdjacencyNeighborhood(center=True)).tolist() == [2, 0]
    assert space._index_table(AdjacencyNeighborhood(center=True)).shape == (4, 4)
