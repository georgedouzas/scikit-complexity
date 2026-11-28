"""Test degrees, mixing, centralities, communities, spectra, neighbour degrees, and summaries of networks."""

import math

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import NetworkMeasuresResult, network_measures
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.mathematics.networks import (
    BarabasiAlbertSystem,
    ConfigurationModelSystem,
    NetworkDiffusionSystem,
)
from skcomplex.systems.social.segregation import SchellingSystem


def _network(kind: str, source: object, **settings: object) -> NetworkMeasuresResult:
    """Return the network analysis of a source for one kind, run."""
    return network_measures(source, kind=kind, **settings)


def _schelling_contacts() -> nx.Graph:
    """Return the network of neighbouring residents of a relaxed Schelling system, with their groups."""
    trajectory = SchellingSystem(width=30, height=30, random_state=0).prepare().simulate(until=100, random_state=0)
    residents = trajectory.get_agents('residents', -1)
    cells = {tuple(int(item) for item in place): index for index, place in enumerate(residents.locations)}
    group = next(iter(residents.values.values()))
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


def _bridged_cliques() -> nx.Graph:
    """Return two complete graphs of 5 nodes joined by the edge from node 4 to node 5."""
    graph = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    graph.add_edge(4, 5)
    return graph


def _small_digraph() -> nx.DiGraph:
    """Return the directed graph with edges 0 to 1, 1 to 0, 1 to 2, and 2 to 0."""
    return nx.DiGraph([(0, 1), (1, 0), (1, 2), (2, 0)])


def _planted_partition(blocks: int, size: int, inside: float, outside: float, seed: int) -> nx.Graph:
    """Return a stochastic block system with the mean in-block and out-block degrees given."""
    probabilities = [
        [inside / (size - 1) if row == column else outside / ((blocks - 1) * size) for column in range(blocks)]
        for row in range(blocks)
    ]
    return nx.stochastic_block_model([size] * blocks, probabilities, seed=seed)


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
    """Return frames of an adaptive voter system that rewires discordant links with probability 0.9."""
    graph = nx.gnm_random_graph(200, 400, seed=1)
    nx.set_node_attributes(graph, {node: int(rng.integers(2)) for node in graph}, 'state')
    frames = []
    for step in range(20_000):
        _adaptive_step(graph, rng)
        if step % 2000 == 0:
            frames.append(graph.copy())
    frames.append(graph.copy())
    return frames


@pytest.mark.slow
def test_summarize_network_small_world():
    """Test that rewiring 1 percent of a ring lattice keeps its clustering and shortens its paths."""
    regular = _network('summary', nx.watts_strogatz_graph(1000, 10, 0.0, seed=0))
    small = _network('summary', nx.watts_strogatz_graph(1000, 10, 0.01, seed=0))
    assert small.clustering[0] > 0.8 * regular.clustering[0]
    assert small.path_length[0] < 0.3 * regular.path_length[0]
    assert _network('degrees', nx.watts_strogatz_graph(1000, 10, 0.0, seed=0)).moments[0].tolist() == [10.0, 100.0]


def test_centrality_of_a_star():
    """Test that the hub of a star of 11 nodes has normalized betweenness 1 and every leaf 0."""
    betweenness = _network('centrality', nx.star_graph(10), measure='betweenness').centrality[0]
    assert betweenness[0] == 1
    assert all(betweenness[leaf] == 0 for leaf in range(1, 11))


def test_assortativity_attribute_mixing_on_two_domains():
    """Test that Schelling neighbours and block-system nodes mix by state, unlike a random relabelling."""
    rng = np.random.default_rng(0)
    contacts = _schelling_contacts()
    blocks = _planted_partition(2, 500, 10.0, 1.0, 1)
    nx.set_node_attributes(blocks, {node: blocks.nodes[node]['block'] for node in blocks}, 'state')
    for graph in (contacts, blocks):
        assert _network('assortativity', graph, attribute='state').assortativity[0] > 0.5
        states = [graph.nodes[node]['state'] for node in graph]
        nx.set_node_attributes(graph, dict(zip(graph.nodes, rng.permutation(states), strict=True)), 'state')
        assert _network('assortativity', graph, attribute='state').assortativity[0] == pytest.approx(0, abs=0.05)


def test_neighbor_degree_and_degree_assortativity():
    """Test the mean neighbour degree of a star and the disassortativity of its degrees."""
    found = _network('neighbor-degree', nx.star_graph(5))
    assert found.distinct_degrees[0].tolist() == [1, 5]
    assert found.neighbor_degree[0].tolist() == [5.0, 1.0]
    assert _network('assortativity', nx.star_graph(5)).assortativity[0] == pytest.approx(-1)


def test_communities_planted_partition():
    """Test that 4 blocks of 100 nodes with in-block degree 16 and out-block degree 2 are recovered."""
    graph = _planted_partition(4, 100, 16.0, 2.0, 2)
    found = _network('communities', graph, reference='block', random_state=0)
    assert found.agreement[0] > 0.9
    assert found.modularity[0] > 0.5
    propagation = _network('communities', graph, method='label_propagation', reference='block')
    assert propagation.agreement[0] > 0.5
    assert _network('communities', graph, random_state=0).agreement is None
    with pytest.raises(AnalysisError, match='louvain, label_propagation'):
        network_measures(np.zeros(10), kind='communities', method='spectral')


@pytest.mark.slow
def test_degrees_of_a_growing_network():
    """Test that frames of a preferential attachment network give a degree exponent near 3 on the last frame."""
    graph = nx.barabasi_albert_graph(10_000, 2, seed=0)
    frames = [graph.subgraph(range(size)).copy() for size in range(1000, 10_001, 1000)]
    found = [_network('degrees', frame) for frame in frames]
    assert _network('summary', frames[-1]).nodes[0] == 10_000
    assert found[-1].exponent[0] == pytest.approx(3, abs=0.2)


def test_assortativity_of_adaptive_rewiring():
    """Test that the assortativity by state of an adaptive voter system rises and ends above 0.9."""
    frames = _adaptive_voter_frames(np.random.default_rng(4))
    values = np.asarray([_network('assortativity', frame, attribute='state').assortativity[0] for frame in frames])
    assert values[-1] > values[0]
    assert values[-1] > 0.9


def test_small_world_measures():
    """Test that rewiring a ring lattice shortens paths while keeping much of its clustering."""
    ring = _network('summary', nx.watts_strogatz_graph(500, 10, 0.0, seed=0))
    small_world = _network('summary', nx.watts_strogatz_graph(500, 10, 0.05, seed=0))
    assert small_world.path_length[0] < ring.path_length[0] / 2
    assert small_world.clustering[0] > 0.5 * ring.clustering[0]
    assert ring.components == [[500]]


def test_centralities():
    """Test that the hub of a star has the largest centrality of every kind."""
    star = nx.star_graph(5)
    for measure in ('degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank'):
        values = _network('centrality', star, measure=measure).centrality[0]
        assert max(values, key=values.get) == 0
    with pytest.raises(AnalysisError, match='Use one of degree'):
        network_measures(np.zeros(10), kind='centrality', measure='fame')


def test_network_analysis_of_every_frame_of_a_trajectory():
    """Test that a network trajectory gives one summary per frame and that an ignored setting raises an error."""
    trajectory = NetworkDiffusionSystem().prepare().simulate(until=0.2, random_state=0)
    summary = _network('summary', trajectory)
    assert summary.nodes.tolist() == [34] * len(trajectory.times)
    with pytest.raises(AnalysisError, match='does not read measure'):
        network_measures(np.zeros(10), kind='summary', measure='pagerank')


def test_network_analysis_of_a_growing_network_reads_every_frame():
    """Test that the summary of a Barabási-Albert trajectory follows its growth and measures its last frame."""
    trajectory = BarabasiAlbertSystem().prepare().simulate(until=50, dt=10, random_state=1)
    summary = _network('summary', trajectory)
    last = trajectory.get_network(-1).get_graph()
    assert summary.nodes.tolist() == [3, 13, 23, 33, 43, 53]
    assert summary.edges[-1] == last.number_of_edges() == 103
    assert summary.clustering[-1] == pytest.approx(nx.average_clustering(last))


def test_summary_of_chosen_measures():
    """Test that the summary computes only the chosen measures and rejects an unknown one."""
    found = _network('summary', nx.karate_club_graph(), measures=['clustering', 'transitivity'])
    assert found.clustering is not None
    assert found.transitivity is not None
    assert found.path_length is None
    assert found.diameter is None
    assert found.components is None
    with pytest.raises(AnalysisError, match='Use one of size, clustering, transitivity, paths'):
        _network('summary', nx.karate_club_graph(), measures=['radius'])


def test_transitivity_against_average_clustering():
    """Test the average clustering and the transitivity of a triangle with 10 leaves on one corner."""
    graph = nx.Graph([(0, 1), (1, 2), (0, 2)] + [(0, leaf) for leaf in range(3, 13)])
    found = _network('summary', graph)
    assert found.clustering[0] == pytest.approx(0.155, abs=1e-3)
    assert found.transitivity[0] == pytest.approx(0.044, abs=1e-3)
    assert np.isnan(found.reciprocity[0])


def test_sampled_paths_of_a_large_graph():
    """Test that sampled sources estimate the mean path length within 2 percent and bound the diameter from below."""
    graph = ConfigurationModelSystem(n_nodes=1000, exponent=2.5).prepare().system_.space.graph
    exact = _network('summary', graph, measures=['paths'])
    sampled = _network('summary', graph, measures=['paths'], samples=200, random_state=0)
    assert sampled.path_length[0] == pytest.approx(exact.path_length[0], rel=0.02)
    assert sampled.diameter[0] <= exact.diameter[0]
    again = _network('summary', graph, measures=['paths'], samples=50, random_state=3)
    repeated = _network('summary', graph, measures=['paths'], samples=50, random_state=3)
    assert again.path_length[0] == repeated.path_length[0]
    assert again.diameter[0] == repeated.diameter[0]


def test_degrees_and_reciprocity_of_a_directed_graph():
    """Test the in and out degrees, their equal first moments, and the reciprocity of a small directed graph."""
    graph = _small_digraph()
    found = _network('degrees', graph)
    assert found.in_degrees[0].tolist() == [2, 1, 1]
    assert found.out_degrees[0].tolist() == [1, 2, 1]
    assert _network('summary', graph).reciprocity[0] == pytest.approx(0.5)
    random = nx.gnp_random_graph(60, 0.1, seed=0, directed=True)
    moments = _network('degrees', random)
    expected = random.number_of_edges() / random.number_of_nodes()
    assert moments.in_moments[0][0] == pytest.approx(expected)
    assert moments.out_moments[0][0] == pytest.approx(expected)
    assert _network('degrees', nx.path_graph(3)).in_degrees is None


def test_spectra_of_small_graphs():
    """Test the Laplacian of a cycle, the connectivity of a path, the radius of a complete graph, and a split graph."""
    cycle = _network('spectrum', nx.cycle_graph(4))
    np.testing.assert_allclose(cycle.laplacian_spectrum[0], [0, 2, 2, 4], atol=1e-12)
    path = _network('spectrum', nx.path_graph(4))
    assert path.algebraic_connectivity[0] == pytest.approx(2 - 2 * math.cos(math.pi / 4), abs=1e-12)
    assert np.count_nonzero(np.diff(np.sign(path.fiedler_vector[0]))) == 1
    complete = _network('spectrum', nx.complete_graph(5))
    assert complete.spectral_radius[0] == pytest.approx(4, abs=1e-12)
    assert 1 / complete.spectral_radius[0] == pytest.approx(0.25)
    split = _network('spectrum', nx.Graph([(0, 1), (2, 3)]))
    assert split.algebraic_connectivity[0] == 0
    assert np.isnan(split.fiedler_vector[0]).all()
    matrix = _network('spectrum', nx.karate_club_graph()).modularity_matrix[0]
    np.testing.assert_allclose(matrix.sum(axis=1), 0, atol=1e-12)


def test_weighted_and_unweighted_louvain():
    """Test that Louvain reads the edge weights of the karate club by default and ignores them with weight None."""
    club = nx.karate_club_graph()
    weighted = _network('communities', club, random_state=0)
    unweighted = _network('communities', club, random_state=0, weight=None)
    assert weighted.modularity[0] == pytest.approx(0.4439, abs=1e-4)
    assert unweighted.modularity[0] == pytest.approx(0.4198, abs=1e-4)


def test_modularity_of_a_given_partition():
    """Test the modularity of the two factions of the karate club and the error without a partition."""
    found = _network('modularity', nx.karate_club_graph(), partition='club', weight=None)
    assert found.modularity[0] == pytest.approx(0.3582, abs=1e-4)
    assert set(found.labels[0].tolist()) == {0, 1}
    with pytest.raises(AnalysisError, match='Give partition'):
        _network('modularity', nx.karate_club_graph())


def test_girvan_newman_communities():
    """Test that Girvan-Newman cuts a bridge and picks the split of highest modularity of the karate club."""
    bridge = _network('communities', _bridged_cliques(), method='girvan_newman', n_communities=2)
    assert bridge.labels[0].tolist() == [0] * 5 + [1] * 5
    club = _network('communities', nx.karate_club_graph(), method='girvan_newman', weight=None)
    assert len(set(club.labels[0].tolist())) == 5
    assert club.modularity[0] == pytest.approx(0.401, abs=1e-3)


def test_leading_eigenvector_communities():
    """Test that the leading eigenvector method splits the karate club above modularity 0.39 and repeats itself."""
    club = nx.karate_club_graph()
    first = _network('communities', club, method='leading_eigenvector', weight=None)
    second = _network('communities', club, method='leading_eigenvector', weight=None)
    assert first.modularity[0] > 0.39
    assert first.labels[0].tolist() == second.labels[0].tolist()
    with pytest.raises(AnalysisError, match=r'does not read n_communities\. The methods girvan_newman'):
        _network('communities', club, method='louvain', n_communities=2)


def test_katz_centrality():
    """Test the Katz centrality of a star against NetworkX, its default attenuation, and a divergent one."""
    star = nx.star_graph(4)
    found = _network('centrality', star, measure='katz', alpha=0.1).centrality[0]
    expected = nx.katz_centrality_numpy(star, alpha=0.1, beta=1.0)
    assert all(found[node] == pytest.approx(expected[node], abs=1e-12) for node in star)
    assert max(found, key=found.get) == 0
    default = _network('centrality', star, measure='katz').centrality[0]
    reference = nx.katz_centrality_numpy(star, alpha=0.9 / 2, beta=1.0)
    assert all(default[node] == pytest.approx(reference[node], abs=1e-12) for node in star)
    with pytest.raises(AnalysisError, match='inverse spectral radius'):
        _network('centrality', star, measure='katz', alpha=0.5)
    with pytest.raises(AnalysisError, match='does not read alpha'):
        _network('centrality', star, alpha=0.1)


def test_edge_betweenness_of_a_bridge():
    """Test that the edge joining two complete graphs has the largest edge betweenness."""
    found = _network('centrality', _bridged_cliques(), measure='edge_betweenness').centrality[0]
    assert max(found, key=found.get) == (4, 5)
