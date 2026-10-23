"""Test the `_networks` module of `skcomplex.models.physics.synchronization`."""

import numpy as np
import pytest

from skcomplex.core.compile import validate
from skcomplex.models.physics.synchronization import coupled_map_lattices_on_networks, kuramoto_on_networks

from ..conftest import _run_observed


@pytest.mark.parametrize(('K', 'synchronized'), [(0.3, False), (3.0, True)])
def test_kuramoto_on_networks_synchronizes_above_the_critical_coupling(K, synchronized):
    """Test that oscillators on a small-world network stay incoherent at weak coupling and synchronize at strong."""
    order = np.asarray(
        _run_observed('kuramoto_on_networks', kuramoto_on_networks(200, K=K), 40.0, 0.5)['order'],
    )
    assert order[0] < 0.15
    assert bool(order[-10:].mean() > 0.8) is synchronized


@pytest.mark.parametrize('network', ['watts_strogatz', 'erdos_renyi', 'barabasi_albert'])
def test_kuramoto_on_networks_builds_its_network(network):
    """Test that every generator gives a valid system with one edge member per edge."""
    system = kuramoto_on_networks(40, network=network, degree=4)
    space = system.components[0].space
    assert system.components[1].size == space.graph.number_of_edges()
    assert validate(system).ok


def test_kuramoto_on_networks_rejects_an_unknown_network():
    """Test that an unknown generator raises an error naming the generators."""
    with pytest.raises(ValueError, match='watts_strogatz'):
        kuramoto_on_networks(10, network='lattice')


@pytest.mark.parametrize(('epsilon', 'synchronized'), [(0.2, False), (0.9, True)])
def test_coupled_maps_on_networks_synchronize_at_strong_coupling(epsilon, synchronized):
    """Test that chaotic logistic maps on a network fall onto one orbit at strong coupling and stay apart at weak."""
    system = coupled_map_lattices_on_networks(100, epsilon=epsilon)
    spread = np.asarray(
        _run_observed('coupled_map_lattices_on_networks', system, 200.0, 1.0)['spread'],
    )
    assert spread[0] > 0.05
    assert bool(spread[-10:].mean() < 1e-6) is synchronized
