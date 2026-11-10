"""Test the `_edge_draw` module of `skcomplex.systems.biology.common.networks`."""

import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.common.networks import EdgeDrawInteraction, WeightedEdgeElement


def _weights(noise, fresh, until=3):
    """Return the weights of the edges of a cycle at every step, under a draw with some noise."""
    graph = nx.cycle_graph(6, create_using=nx.DiGraph)
    space = NetworkSpace(graph)
    nodes = GroupElement('nodes', element=Element, space=space, on='sites')
    edges = GroupElement(
        'edges',
        element=WeightedEdgeElement,
        space=space,
        on='edges',
        values={'w': np.full(6, 5.0), 'fresh': fresh},
    )
    draw = EdgeDrawInteraction('draw', target='nodes', edges='edges', noise=noise)
    system = CompositeSystem(elements=[nodes, edges], interactions=[draw], time=DiscreteTime())
    trajectory = simulate(system, random_state=0, until=until)
    return np.asarray(trajectory.get_values('edges.w'))


def test_edge_draw_draws_new_weights_once_in_the_range():
    """Test that fresh edges draw their weight once in [-1, 1] and keep it without noise."""
    weights = _weights(0.0, fresh=1.0)
    assert np.all(np.abs(weights[1]) <= 1)
    assert np.allclose(weights[1], weights[-1])


def test_edge_draw_drifts_the_weights_of_old_edges():
    """Test that the weights of edges that are not fresh drift around their value with noise."""
    weights = _weights(0.1, fresh=0.0)
    assert not np.allclose(weights[1], weights[0])
    assert np.all(np.abs(weights[-1] - 5.0) < 1.5)
