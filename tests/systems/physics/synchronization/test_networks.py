"""Test the `_networks` module of `skcomplex.systems.physics.synchronization`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.synchronization import (
    CoupledMapLatticesOnNetworksSystem,
    KuramotoOnNetworksSystem,
    LogisticCouplingInteraction,
    LogisticNodeElement,
)

from ..conftest import _run_observed


def test_logistic_coupling_mixes_the_map_with_the_mean_map_of_the_neighbours():
    """Test that one step gives `(1 - epsilon) f(x_i) + epsilon mean_j f(x_j)`, and `(1 - epsilon) f(x_i)` alone."""
    graph = nx.path_graph(3)
    graph.add_node(3)
    x = np.array([0.1, 0.3, 0.6, 0.2])
    maps = GroupElement(
        'maps',
        element=LogisticNodeElement,
        space=NetworkSpace(graph),
        values={'x': x, 'r': 3.7, 'epsilon': 0.3},
        on='sites',
    )
    system = CompositeSystem(
        elements=[maps],
        interactions=[LogisticCouplingInteraction('coupling', target='maps')],
        time=DiscreteTime(),
    )
    values = simulate(system, until=1).get_values('maps.x')[-1]
    f = 3.7 * x * (1 - x)
    neighbours = np.array([f[1], (f[0] + f[2]) / 2, f[1], 0.0])
    np.testing.assert_allclose(values, 0.7 * f + 0.3 * neighbours, atol=1e-12)


@pytest.mark.parametrize(('K', 'synchronized'), [(0.3, False), (3.0, True)])
def test_kuramoto_on_networks_synchronizes_above_the_critical_coupling(K, synchronized):
    """Test that oscillators on a small-world network stay incoherent at weak coupling and synchronize at strong."""
    order = np.asarray(
        _run_observed(KuramotoOnNetworksSystem(200, K=K), 40.0, 0.5).get_values('order'),
    )
    assert order[0] < 0.15
    assert bool(order[-10:].mean() > 0.8) is synchronized


@pytest.mark.parametrize('network', ['watts_strogatz', 'erdos_renyi', 'barabasi_albert'])
def test_kuramoto_on_networks_builds_its_network(network):
    """Test that every generator gives a valid system with one edge member per edge."""
    system = KuramotoOnNetworksSystem(40, network=network, degree=4).prepare().system_
    space = system.elements[0].space
    assert system.elements[1].prepare().size_ == space.graph.number_of_edges()
    assert PassCompiler().prepare(system).report_.ok


def test_kuramoto_on_networks_rejects_an_unknown_network():
    """Test that an unknown generator raises an error naming the generators."""
    with pytest.raises(ValueError, match='watts_strogatz'):
        KuramotoOnNetworksSystem(10, network='lattice').prepare()


@pytest.mark.parametrize(('epsilon', 'synchronized'), [(0.2, False), (0.9, True)])
def test_coupled_maps_on_networks_synchronize_at_strong_coupling(epsilon, synchronized):
    """Test that chaotic logistic maps on a network fall onto one orbit at strong coupling and stay apart at weak."""
    system = CoupledMapLatticesOnNetworksSystem(100, epsilon=epsilon)
    spread = np.asarray(
        _run_observed(system, 200.0, 1.0).get_values('spread'),
    )
    assert spread[0] > 0.05
    assert bool(spread[-10:].mean() < 1e-6) is synchronized
