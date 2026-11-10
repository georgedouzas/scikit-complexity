"""Test the `_structure` module of `skcomplex.systems.mathematics.networks`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.systems.mathematics.networks import (
    NetworkStructureAnalysisSystem,
)

from ..conftest import _run


def test_network_structure_analysis_of_a_generated_network():
    """Test that a planted partition of 4 blocks of 100 nodes is recorded with its measures and its blocks found."""
    trajectory = _run(
        NetworkStructureAnalysisSystem(null_samples=5),
        1,
    )
    assert trajectory.get_agents('nodes', 0).ids.size == 400
    for name in ('clustering', 'path_length', 'assortativity', 'modularity', 'triangle_zscore'):
        assert np.isfinite(trajectory.get_values(name)[-1])
    assert trajectory.get_values('agreement')[-1] > 0.9
    assert len(np.unique(trajectory.get_agents('nodes', 0).values['community'])) == 4


def test_network_structure_analysis_of_a_given_graph():
    """Test that Zachary's karate club gives the NetworkX clustering and path length and at least two communities."""
    graph = nx.karate_club_graph()
    trajectory = _run(
        NetworkStructureAnalysisSystem(graph, reference='club', null_samples=0),
        1,
    )
    assert trajectory.get_values('clustering')[-1] == pytest.approx(nx.average_clustering(graph), abs=1e-12)
    assert trajectory.get_values('path_length')[-1] == pytest.approx(nx.average_shortest_path_length(graph), abs=1e-12)
    assert len(np.unique(trajectory.get_agents('nodes', 0).values['community'])) >= 2
    assert trajectory.get_values('modularity')[-1] > 0.35
    assert trajectory.get_values('triangle_zscore')[-1] == 0.0
