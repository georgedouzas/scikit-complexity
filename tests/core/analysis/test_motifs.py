"""Test motif counts of three and four nodes and their z-scores against null ensembles."""

import itertools

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import MotifsResult, motifs
from skcomplex.core.errors import AnalysisError


def _motifs(kind: str, source: object, **settings: object) -> MotifsResult:
    """Return the motif analysis of a source for one kind, run."""
    return motifs(source, kind=kind, **settings)


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
    scores = _motifs('z-scores', graph, samples=100, random_state=0).z_scores[0]
    assert scores['030T'] > 3
    repeated = _motifs('z-scores', graph, samples=100, random_state=0).z_scores[0]
    np.testing.assert_array_equal(list(repeated.values()), list(scores.values()))
    configuration = _motifs('z-scores', graph, null='configuration', samples=20, random_state=0)
    assert configuration.z_scores[0]['030T'] > 3


def test_motif_counts_of_three_nodes_match_the_triadic_census():
    """Test that directed triads match the triadic census and undirected ones the triangles and wedges."""
    directed = nx.gnm_random_graph(40, 120, seed=1, directed=True)
    census = {name: count for name, count in nx.triadic_census(directed).items() if name not in {'003', '012', '102'}}
    assert _motifs('counts', directed).counts == [census]
    undirected = nx.gnm_random_graph(40, 120, seed=1)
    triangles = sum(nx.triangles(undirected).values()) // 3
    wedges = sum(degree * (degree - 1) // 2 for _, degree in undirected.degree())
    assert _motifs('counts', undirected).counts == [{'path': wedges - 3 * triangles, 'triangle': triangles}]


def test_motif_counts_of_four_nodes_sum_to_the_connected_sets():
    """Test that the four-node motifs of a random graph add up to its connected sets of four nodes."""
    graph = nx.gnm_random_graph(15, 30, seed=2)
    connected = sum(nx.is_connected(graph.subgraph(nodes)) for nodes in itertools.combinations(graph.nodes(), 4))
    assert sum(_motifs('counts', graph, size=4).counts[0].values()) == connected


def test_motif_counts_of_four_nodes():
    """Test the four-node motifs of a clique, a path, a star, and a directed cycle."""
    assert _motifs('counts', nx.complete_graph(4), size=4).counts == [{'clique': 1}]
    assert _motifs('counts', nx.path_graph(4), size=4).counts == [{'path': 1}]
    assert _motifs('counts', nx.star_graph(3), size=4).counts == [{'star': 1}]
    assert _motifs('counts', nx.cycle_graph(5), size=4).counts == [{'path': 5}]
    directed = nx.DiGraph(nx.cycle_graph(4, create_using=nx.DiGraph))
    assert sum(_motifs('counts', directed, size=4).counts[0].values()) == 1
    with pytest.raises(AnalysisError, match='size=5'):
        motifs(np.zeros(10), size=5)
    with pytest.raises(AnalysisError, match="null='shuffle'"):
        motifs(np.zeros(10), kind='z-scores', null='shuffle')
    with pytest.raises(AnalysisError, match='does not read samples'):
        motifs(np.zeros(10), kind='counts', samples=10)
