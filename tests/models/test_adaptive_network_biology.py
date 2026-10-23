"""Test the biology models on fixed and changing networks: epidemics, ecosystems, and autocatalytic sets."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.models import catalog, create

MODELS = {
    'biology/epidemiology': {'quenched_mean_field_sis', 'adaptive_sis', 'activity_driven_epidemics'},
    'biology/ecology': {'sequentially_linear_ecosystem'},
    'biology/evolution': {'jain_krishna', 'sole_manrubia'},
}


def _run(name, until, dt, **arguments):
    """Return the trajectory of a model created by name and run with a fixed seed."""
    return Simulation(create(name, **arguments), dt=dt, random_state=1).run(until=until).trajectory_


@pytest.mark.parametrize(('field', 'names'), list(MODELS.items()))
def test_catalog_lists_the_network_models(field, names):
    """Test that the catalog lists the network models of every subfield with descriptions and references."""
    entries = {entry.name: entry for entry in catalog(field=field)}
    assert names <= set(entries)
    assert all(entries[name].description and entries[name].references for name in names)


def test_quenched_mean_field_threshold_is_the_inverse_spectral_radius():
    """Test that the infection dies out below the inverse of the largest adjacency eigenvalue and persists above."""
    graph = create('quenched_mean_field_sis').elements[0].space.graph
    threshold = 1 / max(np.linalg.eigvalsh(nx.to_numpy_array(graph)))
    below = _run('quenched_mean_field_sis', 40, 0.5, lam=0.5 * threshold)['nodes.p'][-1]
    above = _run('quenched_mean_field_sis', 40, 0.5, lam=2 * threshold)['nodes.p'][-1]
    assert below.max() < 1e-6
    assert above.mean() > 0.1


def test_rewiring_lowers_the_prevalence_of_adaptive_sis():
    """Test that rewiring away from infected neighbours lowers the prevalence and keeps the number of edges."""
    static = _run('adaptive_sis', 30, 1.0, w=0.0)
    adaptive = _run('adaptive_sis', 30, 1.0, w=4.0)
    prevalence = [np.nanmean(item['hosts.i'][10:]) for item in (static, adaptive)]
    assert prevalence[0] > 0.3
    assert prevalence[1] < 0.5 * prevalence[0]
    assert adaptive.network(-1).sources.size == adaptive.network(0).sources.size


def test_activity_driven_edges_last_one_step_and_spread_above_threshold():
    """Test that the edges of a step vanish at the next step, and the infection spreads only above threshold."""
    above = _run('activity_driven_epidemics', 100, 1.0, lam=0.9)
    below = _run('activity_driven_epidemics', 100, 1.0, lam=0.05)
    assert np.nanmean(above['hosts.i'][-20:]) > 0.1
    assert np.nanmean(below['hosts.i'][-1]) == 0.0
    first, second = above.network(50), above.network(51)
    assert first.sources.size > 0
    assert not set(first.edges.tolist()) & set(second.edges.tolist())


def test_sole_manrubia_keeps_its_species_and_extinguishes_in_avalanches():
    """Test that extinct species are replaced, and extinctions come in avalanches of several species."""
    trajectory = _run('sole_manrubia', 200, 1.0)
    sizes = np.nansum(trajectory['species.age'][1:] == 0, axis=1)
    assert all(trajectory.network(frame).nodes.size == 100 for frame in (0, 100, 200))
    assert sizes.sum() > 50
    assert sizes[100:].max() > 1
    assert np.count_nonzero(sizes[100:] == 0) > 10


def test_jain_krishna_grows_an_autocatalytic_set_that_spans_the_species():
    """Test that the populated species jump from a few to nearly all once an autocatalytic set forms."""
    trajectory = _run('jain_krishna', 1500, 5.0)
    populated = np.nansum(trajectory['species.x'] > 1e-3, axis=1)
    links = [trajectory.network(frame).sources.size for frame in (1, len(trajectory) - 1)]
    assert populated[1:40].min() < 20
    assert populated[-1] >= 35
    assert links[1] > links[0]


def test_sequentially_linear_ecosystem_settles_after_bursts_of_extinctions():
    """Test that abundances stay normalised and extinctions grow rarer as the network settles."""
    trajectory = _run('sequentially_linear_ecosystem', 300, 1.0)
    extinctions = np.nansum(trajectory['species.age'][1:] == 0, axis=1)
    assert np.allclose(np.nansum(trajectory['species.x'][-1]), 1.0, atol=0.05)
    assert extinctions[:100].sum() > extinctions[200:].sum()
    assert all(trajectory.network(frame).nodes.size == 60 for frame in (0, 150, 300))
