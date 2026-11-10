"""Test the `_tipping` module of `skcomplex.systems.physics.climate`."""

import networkx as nx
import numpy as np

from skcomplex.systems.physics.climate import TippingCascadeNetworkSystem

from ..conftest import _run_default


def test_tipping_cascade_network_tips_alone_only_beyond_the_critical_control():
    """Test that an uncoupled element tips only once its control passes the critical value 1."""
    pair = nx.path_graph(2)
    below = _run_default(TippingCascadeNetworkSystem(pair, c=[0.95, 0.0], d=0.0), until=100.0, dt=1.0).get_values(
        'elements.x',
    )[-1]
    above = _run_default(TippingCascadeNetworkSystem(pair, c=[1.05, 0.0], d=0.0), until=100.0, dt=1.0).get_values(
        'elements.x',
    )[-1]
    assert below[0] < 0
    assert above[0] > 1


def test_tipping_cascade_network_cascades_to_neighbours():
    """Test that an element pushed past its tipping point tips coupled neighbours that would not tip alone."""
    graph = nx.path_graph(4)
    controls = [1.1, 0.3, 0.3, 0.3]
    alone = _run_default(TippingCascadeNetworkSystem(graph, c=controls, d=0.0), until=60.0, dt=1.0).get_values(
        'elements.x',
    )[-1]
    coupled = _run_default(TippingCascadeNetworkSystem(graph, c=controls, d=0.2), until=60.0, dt=1.0).get_values(
        'elements.x',
    )[-1]
    held = _run_default(
        TippingCascadeNetworkSystem(graph, c=[0.9, 0.3, 0.3, 0.3], d=0.2),
        until=60.0,
        dt=1.0,
    ).get_values('elements.x')[-1]
    assert alone[0] > 0
    assert np.all(alone[1:] < 0)
    assert np.all(coupled > 0)
    assert np.all(held < 0)
