"""Test degrees, mixing, centralities, communities, neighbour degrees, and summaries of networks."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import NetworkMeasuresResult, network_measures
from skcomplex.core.errors import AnalysisError
from skcomplex.systems import create_simulation


def _network(kind: str, source: object, **settings: object) -> NetworkMeasuresResult:
    """Return the network analysis of a source for one kind, run."""
    return network_measures(source, kind=kind, **settings)


def _schelling_contacts() -> nx.Graph:
    """Return the network of neighbouring residents of a relaxed Schelling system, with their groups."""
    trajectory = (
        create_simulation('schelling', width=30, height=30, random_state=0).prepare().run(until=100).trajectory_
    )
    residents = trajectory.agents('residents', -1)
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
    trajectory = create_simulation('network_diffusion').prepare().run(until=0.2).trajectory_
    summary = _network('summary', trajectory)
    assert summary.nodes.tolist() == [34] * len(trajectory.times)
    with pytest.raises(AnalysisError, match='does not read measure'):
        network_measures(np.zeros(10), kind='summary', measure='pagerank')
