"""Test the `_adaptive` module of `skcomplex.models.biology.epidemiology`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.models import catalog, create

from ..conftest import _run

MODELS = {
    'biology/epidemiology': {'quenched_mean_field_sis', 'adaptive_sis', 'activity_driven_epidemics'},
    'biology/ecology': {'sequentially_linear_ecosystem'},
    'biology/evolution': {'jain_krishna', 'sole_manrubia'},
}


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
    below = _run('quenched_mean_field_sis', 40, 0.5, lam=0.5 * threshold).values('nodes.p')[-1]
    above = _run('quenched_mean_field_sis', 40, 0.5, lam=2 * threshold).values('nodes.p')[-1]
    assert below.max() < 1e-6
    assert above.mean() > 0.1


def test_rewiring_lowers_the_prevalence_of_adaptive_sis():
    """Test that rewiring away from infected neighbours lowers the prevalence and keeps the number of edges."""
    static = _run('adaptive_sis', 30, 1.0, w=0.0)
    adaptive = _run('adaptive_sis', 30, 1.0, w=4.0)
    prevalence = [np.nanmean(item.values('hosts.i')[10:]) for item in (static, adaptive)]
    assert prevalence[0] > 0.3
    assert prevalence[1] < 0.5 * prevalence[0]
    assert adaptive.network(-1).sources.size == adaptive.network(0).sources.size


def test_activity_driven_edges_last_one_step_and_spread_above_threshold():
    """Test that the edges of a step vanish at the next step, and the infection spreads only above threshold."""
    above = _run('activity_driven_epidemics', 100, 1.0, lam=0.9)
    below = _run('activity_driven_epidemics', 100, 1.0, lam=0.05)
    assert np.nanmean(above.values('hosts.i')[-20:]) > 0.1
    assert np.nanmean(below.values('hosts.i')[-1]) == 0.0
    first, second = above.network(50), above.network(51)
    assert first.sources.size > 0
    assert not set(first.edges.tolist()) & set(second.edges.tolist())
