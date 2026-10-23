"""Test synchronization on networks and Vicsek flocks with polar, cohesive, and nematic alignment."""

import numpy as np
import pytest

from skcomplex.core.compile import validate
from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.physics.active_matter import ginelli_nematics, gregoire_chate, vicsek
from skcomplex.models.physics.synchronization import coupled_map_lattices_on_networks, kuramoto_on_networks


def _run(name, system, until, dt):
    """Return the trajectory of a model's run with its observables."""
    observables = describe(name).observables_of(system)
    return Simulation(system, dt=dt, random_state=0, observables=observables).run(until=until).trajectory_


@pytest.mark.parametrize(('K', 'synchronized'), [(0.3, False), (3.0, True)])
def test_kuramoto_on_networks_synchronizes_above_the_critical_coupling(K, synchronized):
    """Test that oscillators on a small-world network stay incoherent at weak coupling and synchronize at strong."""
    order = np.asarray(_run('kuramoto_on_networks', kuramoto_on_networks(200, K=K), 40.0, 0.5)['order'])
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
    spread = np.asarray(_run('coupled_map_lattices_on_networks', system, 200.0, 1.0)['spread'])
    assert spread[0] > 0.05
    assert bool(spread[-10:].mean() < 1e-6) is synchronized


def test_vicsek_flocks_at_low_noise_and_disorders_at_high_noise():
    """Test that the polar order of the Vicsek model exceeds 0.9 at low noise and stays small at high noise."""
    ordered = np.asarray(_run('vicsek', vicsek(300), 150.0, 10.0)['polarization'])
    disordered = np.asarray(_run('vicsek', vicsek(300, eta=0.9), 150.0, 10.0)['polarization'])
    assert ordered[0] < 0.2
    assert ordered[-1] > 0.9
    assert disordered[-3:].mean() < 0.3


def test_gregoire_chate_forms_an_aligned_flock():
    """Test that cohesive particles at low noise align into a flock with a polar order above 0.8."""
    polarization = np.asarray(_run('gregoire_chate', gregoire_chate(), 300.0, 10.0)['polarization'])
    assert polarization[0] < 0.2
    assert polarization[-5:].mean() > 0.8


def test_ginelli_nematics_orders_an_axis_without_a_direction():
    """Test that nematic alignment orders the axes of the particles while their polar order stays small."""
    trajectory = _run('ginelli_nematics', ginelli_nematics(), 200.0, 10.0)
    nematic, polar = np.asarray(trajectory['nematic_order']), np.asarray(trajectory['polarization'])
    assert nematic[-1] > 0.85
    assert polar[-1] < 0.3
