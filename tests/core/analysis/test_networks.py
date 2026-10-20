"""Test centralities, mixing, communities, motifs, and measures of evolving networks."""

import functools

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import (
    assortativity,
    centrality,
    communities,
    degree_moments,
    motif_counts,
    motif_zscores,
    neighbor_degree,
    network_frames,
    snapshots_of,
    summarize_network,
)
from skcomplex.core.base import AnalysisError
from skcomplex.models import run


def _schelling_contacts() -> nx.Graph:
    """Return the network of neighbouring residents of a relaxed Schelling model, with their groups."""
    trajectory = run('schelling', width=30, height=30, random_state=0, settings={'until': 100}).trajectory_
    snapshot = snapshots_of(trajectory)
    cells = {tuple(int(item) for item in place): index for index, place in enumerate(snapshot.positions[-1])}
    group = next(values for values in snapshot.states.values())[-1]
    graph = nx.Graph()
    graph.add_nodes_from((index, {'state': int(group[index])}) for index in cells.values())
    shifts = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    graph.add_edges_from(
        (index, cells[(x + dx, y + dy)])
        for (x, y), index in cells.items()
        for dx, dy in shifts
        if (x + dx, y + dy) in cells
    )
    return graph


def _planted_partition(blocks: int, size: int, inside: float, outside: float, seed: int) -> nx.Graph:
    """Return a stochastic block model with the mean in-block and out-block degrees given."""
    probabilities = [
        [inside / (size - 1) if row == column else outside / ((blocks - 1) * size) for column in range(blocks)]
        for row in range(blocks)
    ]
    return nx.stochastic_block_model([size] * blocks, probabilities, seed=seed)


def _feed_forward_graph(rng: np.random.Generator) -> nx.DiGraph:
    """Return a random directed graph of 200 nodes with 50 planted feed-forward loops."""
    graph = nx.gnm_random_graph(200, 400, seed=0, directed=True)
    for _ in range(50):
        first, second, third = rng.choice(200, 3, replace=False)
        graph.add_edges_from([(first, second), (second, third), (first, third)])
    return graph


def _adaptive_step(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Rewire a discordant link to a like node with probability 0.9, or copy the neighbour's state otherwise."""
    edges = list(graph.edges())
    first, second = edges[int(rng.integers(len(edges)))]
    if rng.random() < 0.5:
        first, second = second, first
    state = graph.nodes[first]['state']
    if state == graph.nodes[second]['state']:
        return
    if rng.random() >= 0.9:
        graph.nodes[first]['state'] = graph.nodes[second]['state']
        return
    same = [node for node in graph if graph.nodes[node]['state'] == state]
    target = same[int(rng.integers(len(same)))]
    if target != first and not graph.has_edge(first, target):
        graph.remove_edge(first, second)
        graph.add_edge(first, target)


def _adaptive_voter_frames(rng: np.random.Generator) -> list[nx.Graph]:
    """Return frames of an adaptive voter model that rewires discordant links with probability 0.9."""
    graph = nx.gnm_random_graph(200, 400, seed=1)
    nx.set_node_attributes(graph, {node: int(rng.integers(2)) for node in graph}, 'state')
    frames = []
    for step in range(20_000):
        _adaptive_step(graph, rng)
        if step % 2000 == 0:
            frames.append(graph.copy())
    frames.append(graph.copy())
    return frames


def test_summarize_network_small_world():
    """Test that rewiring 1 percent of a ring lattice keeps its clustering and shortens its paths."""
    regular = summarize_network(nx.watts_strogatz_graph(1000, 10, 0.0, seed=0))
    small = summarize_network(nx.watts_strogatz_graph(1000, 10, 0.01, seed=0))
    assert small.clustering > 0.8 * regular.clustering
    assert small.path_length < 0.3 * regular.path_length
    assert degree_moments(nx.watts_strogatz_graph(1000, 10, 0.0, seed=0)) == [10.0, 100.0]


def test_centrality_of_a_star():
    """Test that the hub of a star of 11 nodes has normalized betweenness 1 and every leaf 0."""
    betweenness = centrality(nx.star_graph(10), 'betweenness')
    assert betweenness[0] == 1
    assert all(betweenness[leaf] == 0 for leaf in range(1, 11))


def test_assortativity_attribute_mixing_on_two_domains():
    """Test that Schelling neighbours and block-model nodes mix by state, unlike a random relabelling."""
    rng = np.random.default_rng(0)
    contacts = _schelling_contacts()
    blocks = _planted_partition(2, 500, 10.0, 1.0, 1)
    nx.set_node_attributes(blocks, {node: blocks.nodes[node]['block'] for node in blocks}, 'state')
    for graph in (contacts, blocks):
        assert assortativity(graph, 'state') > 0.5
        states = [graph.nodes[node]['state'] for node in graph]
        nx.set_node_attributes(graph, dict(zip(graph.nodes, rng.permutation(states), strict=True)), 'state')
        assert assortativity(graph, 'state') == pytest.approx(0, abs=0.05)


def test_neighbor_degree_and_degree_assortativity():
    """Test the mean neighbour degree of a star and the disassortativity of its degrees."""
    degrees, connectivity = neighbor_degree(nx.star_graph(5))
    assert degrees.tolist() == [1, 5]
    assert connectivity.tolist() == [5.0, 1.0]
    assert assortativity(nx.star_graph(5)) == pytest.approx(-1)


def test_communities_planted_partition():
    """Test that 4 blocks of 100 nodes with in-block degree 16 and out-block degree 2 are recovered."""
    graph = _planted_partition(4, 100, 16.0, 2.0, 2)
    found = communities(graph, reference='block', random_state=0)
    assert found.agreement > 0.9
    assert found.modularity > 0.5
    assert communities(graph, 'label_propagation', reference='block').agreement > 0.5
    assert communities(graph, random_state=0).agreement is None
    with pytest.raises(AnalysisError, match='Unknown method'):
        communities(graph, 'spectral')


def test_motif_zscores_feed_forward_loop():
    """Test that planted feed-forward loops score above 3 against edge switches and the same seed repeats."""
    graph = _feed_forward_graph(np.random.default_rng(3))
    scores = motif_zscores(graph, samples=100, random_state=0)
    assert scores.zscores['030T'] > 3
    repeated = motif_zscores(graph, samples=100, random_state=0).zscores
    np.testing.assert_array_equal(list(repeated.values()), list(scores.zscores.values()))
    configuration = motif_zscores(graph, null='configuration', samples=20, random_state=0)
    assert configuration.zscores['030T'] > 3


def test_motif_counts_of_four_nodes():
    """Test the four-node motifs of a clique, a path, a star, and a directed cycle."""
    assert motif_counts(nx.complete_graph(4), 4) == {'clique': 1}
    assert motif_counts(nx.path_graph(4), 4) == {'path': 1}
    assert motif_counts(nx.star_graph(3), 4) == {'star': 1}
    assert motif_counts(nx.cycle_graph(5), 4) == {'path': 5}
    assert sum(motif_counts(nx.DiGraph(nx.cycle_graph(4, create_using=nx.DiGraph)), 4).values()) == 1
    with pytest.raises(AnalysisError, match='3 or 4 nodes'):
        motif_counts(nx.path_graph(3), 5)
    with pytest.raises(AnalysisError, match='Unknown null'):
        motif_zscores(nx.path_graph(3), null='shuffle')


def test_network_frames_growing_network():
    """Test that frames of a preferential attachment run give one row each and a degree exponent near 3."""
    graph = nx.barabasi_albert_graph(10_000, 2, seed=0)
    frames = [graph.subgraph(range(size)).copy() for size in range(1000, 10_001, 1000)]
    rows = network_frames(frames)
    assert len(rows) == len(frames)
    assert rows[-1]['nodes'] == 10_000
    assert rows[-1]['degree_exponent'] == pytest.approx(3, abs=0.2)


def test_network_frames_adaptive_rewiring():
    """Test that the assortativity by state of an adaptive voter model rises and ends above 0.9."""
    frames = _adaptive_voter_frames(np.random.default_rng(4))
    mixing = functools.partial(assortativity, attribute='state')
    values = np.asarray([row['mixing'] for row in network_frames(frames, {'mixing': mixing})])
    assert values[-1] > values[0]
    assert values[-1] > 0.9
