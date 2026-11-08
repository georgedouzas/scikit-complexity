"""Test the `_adaptive` module of `skcomplex.systems.biology.epidemiology`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.systems import all_systems
from skcomplex.systems.biology.epidemiology import (
    ActivityDrivenEpidemicsSystem,
    AdaptiveSISSystem,
    QuenchedMeanFieldSISSystem,
)

from ..conftest import _run

SYSTEMS = {
    'biology/epidemiology': {'QuenchedMeanFieldSISSystem', 'AdaptiveSISSystem', 'ActivityDrivenEpidemicsSystem'},
    'biology/ecology': {'SequentiallyLinearEcosystemSystem'},
    'biology/evolution': {'JainKrishnaSystem', 'SoleManrubiaSystem'},
}


@pytest.mark.parametrize(('field', 'names'), list(SYSTEMS.items()))
def test_all_systems_lists_the_network_systems(field, names):
    """Test that `all_systems` lists the network systems of every subfield with descriptions and references."""
    systems = dict(all_systems(field=field))
    assert names <= set(systems)
    assert all(systems[name].__doc__ and systems[name].references for name in names)


def test_quenched_mean_field_threshold_is_the_inverse_spectral_radius():
    """Test that the infection dies out below the inverse of the largest adjacency eigenvalue and persists above."""
    graph = QuenchedMeanFieldSISSystem().prepare().system_.elements[0].space.graph
    threshold = 1 / max(np.linalg.eigvalsh(nx.to_numpy_array(graph)))
    below = _run(QuenchedMeanFieldSISSystem(lam=0.5 * threshold), 40, 0.5).values('nodes.p')[-1]
    above = _run(QuenchedMeanFieldSISSystem(lam=2 * threshold), 40, 0.5).values('nodes.p')[-1]
    assert below.max() < 1e-6
    assert above.mean() > 0.1


def test_rewiring_lowers_the_prevalence_of_adaptive_sis():
    """Test that rewiring away from infected neighbours lowers the prevalence and keeps the number of edges."""
    static = _run(AdaptiveSISSystem(w=0.0), 30, 1.0)
    adaptive = _run(AdaptiveSISSystem(w=4.0), 30, 1.0)
    prevalence = [np.nanmean(item.values('hosts.i')[10:]) for item in (static, adaptive)]
    assert prevalence[0] > 0.3
    assert prevalence[1] < 0.5 * prevalence[0]
    assert adaptive.network(-1).sources.size == adaptive.network(0).sources.size


def test_activity_driven_edges_last_one_step_and_spread_above_threshold():
    """Test that the edges of a step vanish at the next step, and the infection spreads only above threshold."""
    above = _run(ActivityDrivenEpidemicsSystem(lam=0.9), 100, 1.0)
    below = _run(ActivityDrivenEpidemicsSystem(lam=0.05), 100, 1.0)
    assert np.nanmean(above.values('hosts.i')[-20:]) > 0.1
    assert np.nanmean(below.values('hosts.i')[-1]) == 0.0
    first, second = above.network(50), above.network(51)
    assert first.sources.size > 0
    assert not set(first.edges.tolist()) & set(second.edges.tolist())
