"""Test the `_host_graph` module of `skcomplex.core.elements`."""

import networkx as nx
import pytest

from skcomplex.core.elements import HostGraphElement
from skcomplex.errors import RewritingError


def test_host_graph_takes_a_networkx_graph_or_none():
    """Test that a host graph takes a NetworkX graph or the empty graph, and rejects other values."""
    assert HostGraphElement('ring', graph=nx.cycle_graph(4)).check().graph.number_of_nodes() == 4
    assert HostGraphElement('empty').check().graph is None
    with pytest.raises(RewritingError, match='NetworkX graph'):
        HostGraphElement('edges', graph=[(0, 1)]).check()


def test_host_graph_is_drawn_as_its_geometry():
    """Test that a prepared host graph is drawn as the geometry of the graph."""
    assert HostGraphElement('ring', graph=nx.cycle_graph(4)).check().prepare().default_visual_.kind == 'geometry'
