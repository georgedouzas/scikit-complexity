"""Test the `_adjacency` module of `skcomplex.core.components.spaces`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import AdjacencyNeighborhood, DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import SpaceError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec


def test_adjacency_neighbourhood_puts_the_node_first_when_it_is_included():
    """Test that an adjacency neighbourhood with its centre reads the node before its neighbours."""
    space = NetworkSpace(nx.star_graph(3)).prepare()
    assert space._get_neighbors(2, AdjacencyNeighborhood(center=True)).tolist() == [2, 0]
    assert space._get_index_table(AdjacencyNeighborhood(center=True)).shape == (4, 4)


def test_adjacency_neighbourhood_of_a_missing_layer_lists_the_layers():
    """Test that a neighbourhood that names a missing layer raises a space error that lists the layers."""
    sick = {'variables': {'x': DeclaredVariableSpec(0.0, StateSpec())}, 'x': [1, 0, 0, 0, 0]}
    space = NetworkSpace(nx.path_graph(5))
    people = GroupElement('people', element=DeclaredElement, space=space, on='sites', values=sick)
    spread = LocalRuleInteraction(
        'spread',
        target='people',
        state='x',
        rule=lambda infected, *contacts: np.maximum(infected, sum(contacts) > 0),
        neighborhood=AdjacencyNeighborhood(layer='work'),
    )
    with pytest.raises(SpaceError, match=r"The network has no layer 'work'\. Its layers are None\."):
        simulate(CompositeSystem(elements=[people], interactions=[spread], time=DiscreteTime()), until=2)
    with pytest.raises(SpaceError, match=r"no layer 'work'"):
        NetworkSpace(nx.path_graph(5)).prepare()._get_neighbor_pairs('work')
