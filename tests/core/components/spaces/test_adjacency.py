"""Test the `_adjacency` module of `skcomplex.core.components.spaces`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import AdjacencyNeighborhood, NetworkSpace
from skcomplex.core.errors import SpaceError
from skcomplex.core.systems import System


def test_adjacency_neighbourhood_puts_the_node_first_when_it_is_included():
    """Test that an adjacency neighbourhood with its centre reads the node before its neighbours."""
    space = NetworkSpace(nx.star_graph(3)).prepare()
    assert space._get_neighbors(2, AdjacencyNeighborhood(center=True)).tolist() == [2, 0]
    assert space._get_index_table(AdjacencyNeighborhood(center=True)).shape == (4, 4)


def test_adjacency_neighbourhood_of_a_missing_layer_lists_the_layers():
    """Test that a neighbourhood that names a missing layer raises a space error that lists the layers."""
    space = NetworkSpace(nx.path_graph(5))
    people = Element('people', values={'x': [1.0, 0.0, 0.0, 0.0, 0.0]}, space=space)
    spread = Interaction(
        'spread',
        roles={'target': 'people'},
        laws=lambda s: [
            UpdateLaw(
                s.target.x,
                lambda infected, *contacts, **kwargs: np.maximum(infected, sum(contacts) > 0),
                neighborhood=AdjacencyNeighborhood(layer='work'),
            ),
        ],
    )
    with pytest.raises(SpaceError, match=r"The network has no layer 'work'\. Its layers are None\."):
        System(elements=[people], interactions=[spread]).prepare().simulate(until=2)
    with pytest.raises(SpaceError, match=r"no layer 'work'"):
        NetworkSpace(nx.path_graph(5)).prepare()._get_neighbor_pairs('work')
