"""Test motif counts of three and four nodes and their z-scores against null ensembles."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.analysis import MotifAnalysis
from skcomplex.errors import AnalysisError


def _motifs(kind: str, source: object, **settings: object) -> MotifAnalysis:
    """Return the motif analysis of a source for one kind, run."""
    return MotifAnalysis(kind, **settings).check().prepare(source).run()


def _feed_forward_graph(rng: np.random.Generator) -> nx.DiGraph:
    """Return a random directed graph of 200 nodes with 50 planted feed-forward loops."""
    graph = nx.gnm_random_graph(200, 400, seed=0, directed=True)
    for _ in range(50):
        first, second, third = rng.choice(200, 3, replace=False)
        graph.add_edges_from([(first, second), (second, third), (first, third)])
    return graph


def test_motif_zscores_feed_forward_loop():
    """Test that planted feed-forward loops score above 3 against edge switches and the same seed repeats."""
    graph = _feed_forward_graph(np.random.default_rng(3))
    scores = _motifs('z-scores', graph, samples=100, random_state=0).z_scores_[0]
    assert scores['030T'] > 3
    repeated = _motifs('z-scores', graph, samples=100, random_state=0).z_scores_[0]
    np.testing.assert_array_equal(list(repeated.values()), list(scores.values()))
    configuration = _motifs('z-scores', graph, null='configuration', samples=20, random_state=0)
    assert configuration.z_scores_[0]['030T'] > 3


def test_motif_counts_of_four_nodes():
    """Test the four-node motifs of a clique, a path, a star, and a directed cycle."""
    assert _motifs('counts', nx.complete_graph(4), size=4).counts_ == [{'clique': 1}]
    assert _motifs('counts', nx.path_graph(4), size=4).counts_ == [{'path': 1}]
    assert _motifs('counts', nx.star_graph(3), size=4).counts_ == [{'star': 1}]
    assert _motifs('counts', nx.cycle_graph(5), size=4).counts_ == [{'path': 5}]
    directed = nx.DiGraph(nx.cycle_graph(4, create_using=nx.DiGraph))
    assert sum(_motifs('counts', directed, size=4).counts_[0].values()) == 1
    with pytest.raises(AnalysisError, match='size=5'):
        MotifAnalysis(size=5).check()
    with pytest.raises(AnalysisError, match="null='shuffle'"):
        MotifAnalysis('z-scores', null='shuffle').check()
    with pytest.raises(AnalysisError, match='does not read samples'):
        MotifAnalysis('counts', samples=10).check()
