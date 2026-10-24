"""Test the `_processes` module of `skcomplex.models.mathematics.networks`."""

import networkx as nx
import numpy as np

from skcomplex.models.mathematics.networks import (
    network_diffusion,
    network_random_walk,
    pagerank,
)

from ..conftest import _run


def test_network_random_walk_visits_nodes_in_proportion_to_degree():
    """Test that random walkers occupy the nodes of a connected graph in proportion to their degrees."""
    graph = nx.karate_club_graph()
    trajectory = _run(network_random_walk(graph, n_walkers=200), 100)
    nodes = np.concatenate([trajectory.agents('walkers', frame).locations for frame in range(20, 101)])
    visits = np.bincount(np.ravel(nodes).astype(int), minlength=graph.number_of_nodes())
    degrees = np.array([degree for _, degree in graph.degree()])
    assert np.corrcoef(visits, degrees)[0, 1] > 0.95


def test_network_diffusion_conserves_mass_and_equalizes():
    """Test that diffusion on a connected network conserves the total and tends to the uniform content."""
    contents = _run(network_diffusion(), 40, dt=1.0)['nodes.u']
    np.testing.assert_allclose(contents.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(contents[-1], 1 / 34, atol=1e-4)


def test_pagerank_matches_networkx():
    """Test that the PageRank iteration converges to the PageRank computed by NetworkX."""
    graph = nx.les_miserables_graph()
    ranks = _run(pagerank(graph, damping=0.85), 200)['pages.rank'][-1]
    expected = nx.pagerank(graph, alpha=0.85, weight=None, tol=1e-14, max_iter=1000)
    np.testing.assert_allclose(ranks, [expected[node] for node in graph.nodes], atol=1e-8)
