"""Test the `_logistic_coupling` module of `skcomplex.models.physics.synchronization`."""

import networkx as nx
import numpy as np

from skcomplex.core.elements import LatticeElement
from skcomplex.core.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.synchronization import LogisticCouplingInteraction, LogisticNodeElement


def test_logistic_coupling_mixes_the_map_with_the_mean_map_of_the_neighbours():
    """Test that one step gives `(1 - epsilon) f(x_i) + epsilon mean_j f(x_j)`, and `(1 - epsilon) f(x_i)` alone."""
    graph = nx.path_graph(3)
    graph.add_node(3)
    x = np.array([0.1, 0.3, 0.6, 0.2])
    maps = LatticeElement(
        'maps',
        element=LogisticNodeElement,
        space=NetworkSpace(graph),
        values={'x': x, 'r': 3.7, 'epsilon': 0.3},
    )
    system = System(
        elements=[maps],
        interactions=[LogisticCouplingInteraction('coupling', target='maps')],
        time=DiscreteTime(),
    )
    values = Simulation(system).prepare().run(until=1).trajectory_.values('maps.x')[-1]
    f = 3.7 * x * (1 - x)
    neighbours = np.array([f[1], (f[0] + f[2]) / 2, f[1], 0.0])
    np.testing.assert_allclose(values, 0.7 * f + 0.3 * neighbours, atol=1e-12)
