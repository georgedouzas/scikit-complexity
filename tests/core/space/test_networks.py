"""Test network spaces: generators, layers, and the arrays of their edges and adjacency."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.base import ValidationError
from skcomplex.core.components import Element
from skcomplex.core.laws import UpdateLaw, neighbors
from skcomplex.core.space import Adjacency, Layer, NetworkSpace, neighbor_pairs, part_count
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system

BACKENDS = ('numpy', 'numba')
GENERATORS = [
    lambda seed: NetworkSpace.erdos_renyi(40, 0.1, seed=seed),
    lambda seed: NetworkSpace.watts_strogatz(40, 4, 0.2, seed=seed),
    lambda seed: NetworkSpace.barabasi_albert(40, 2, seed=seed),
    lambda seed: NetworkSpace.configuration_model([3] * 20 + [1] * 20, seed=seed),
    lambda seed: NetworkSpace.stochastic_block_model([20, 20], [[0.3, 0.02], [0.02, 0.3]], seed=seed),
    lambda seed: NetworkSpace.waxman(40, seed=seed),
    lambda seed: NetworkSpace.random_geometric(40, 0.25, seed=seed),
]


def _edges(space):
    """Return the edges of a network space as a sorted list of node index pairs."""
    sources, targets = space.prepare().edges_[None]
    return sorted(zip(sources.tolist(), targets.tolist(), strict=True))


def test_watts_strogatz_with_the_same_seed_gives_the_same_network():
    """Test that a Watts-Strogatz network generated twice with the same seed has identical edges."""
    first = NetworkSpace.watts_strogatz(100, 6, 0.3, seed=7)
    second = NetworkSpace.watts_strogatz(100, 6, 0.3, seed=7)
    assert _edges(first) == _edges(second)
    assert _edges(first) != _edges(NetworkSpace.watts_strogatz(100, 6, 0.3, seed=8))


@pytest.mark.parametrize('generate', GENERATORS)
def test_generators_reproduce_their_network_from_a_random_stream(generate):
    """Test that every random generator reproduces its network from generators of the same random stream."""
    first = generate(np.random.default_rng(3))
    second = generate(np.random.default_rng(3))
    assert first.prepare().size_ == second.prepare().size_
    assert _edges(first) == _edges(second)


def test_deterministic_generators_build_their_graphs():
    """Test that the complete and grid networks have the edges of their graphs and the grid its coordinates."""
    assert NetworkSpace.complete(6).graph.number_of_edges() == 15
    grid = NetworkSpace.grid((3, 4)).prepare()
    assert grid.size_ == 12
    assert grid.positions_[5].tolist() == [1.0, 1.0]
    assert grid.graph.has_edge(5, 6)
    assert grid.graph.has_edge(5, 9)


def test_configuration_model_drops_parallel_edges_and_self_loops():
    """Test that the configuration model returns a simple graph whose degrees do not exceed the given ones."""
    degrees = [4] * 10
    space = NetworkSpace.configuration_model(degrees, seed=0)
    assert not space.graph.is_multigraph()
    assert nx.number_of_selfloops(space.graph) == 0
    assert all(degree <= 4 for _, degree in space.graph.degree)


def test_spatial_generators_store_node_positions():
    """Test that the Waxman and random geometric networks store positions that join near nodes."""
    space = NetworkSpace.random_geometric(60, 0.2, seed=2).prepare()
    positions = space.positions_
    sources, targets = space.edges_[None]
    assert positions.shape == (60, 2)
    assert np.all(np.linalg.norm(positions[sources] - positions[targets], axis=1) <= 0.2)
    assert NetworkSpace.waxman(30, seed=1).prepare().positions_.shape == (30, 2)


def test_nodes_without_a_position_lie_on_a_circle():
    """Test that the nodes of a graph without stored positions get the positions of a circular layout."""
    positions = NetworkSpace(nx.path_graph(4)).prepare().positions_
    assert positions.shape == (4, 2)
    assert np.allclose(np.linalg.norm(positions, axis=1), 1.0)


def test_list_edges_orders_undirected_ends_and_keeps_directions():
    """Test that the source of an undirected edge comes first in node order and a directed edge keeps its direction."""
    undirected = nx.Graph()
    undirected.add_nodes_from(['a', 'b', 'c'])
    undirected.add_edges_from([('c', 'a'), ('b', 'c')])
    assert [item.tolist() for item in NetworkSpace(undirected).prepare().edges_[None]] == [[0, 1], [2, 2]]
    directed = NetworkSpace(nx.DiGraph([(2, 0), (0, 1)])).prepare()
    assert list(directed.graph.nodes) == [2, 0, 1]
    assert [item.tolist() for item in directed.edges_[None]] == [[0, 1], [1, 2]]


def test_compute_adjacency_lists_neighbours_with_their_edges():
    """Test that the adjacency lists every neighbour with the edge that joins it, successors on a directed graph."""
    space = NetworkSpace(nx.Graph([(0, 1), (1, 2), (0, 2)])).prepare()
    indptr, indices, edges = space.adjacency_[None]
    sources, targets = space.edges_[None]
    for node in range(3):
        for neighbour, edge in zip(
            indices[indptr[node] : indptr[node + 1]],
            edges[indptr[node] : indptr[node + 1]],
            strict=True,
        ):
            assert {int(sources[edge]), int(targets[edge])} == {node, int(neighbour)}
    directed = NetworkSpace(nx.DiGraph([(0, 1), (2, 1)])).prepare()
    indptr, indices, _ = directed.adjacency_[None]
    assert np.diff(indptr).tolist() == [1, 0, 1]


def test_layers_hold_relations_over_the_same_nodes():
    """Test that a network space resolves the graph and the edges of every layer by its label."""
    work = Layer('work', nx.complete_graph(4))
    space = NetworkSpace(nx.path_graph(4), layers=[work]).prepare()
    assert list(space.graphs_) == [None, 'work']
    assert space.graphs_['work'] is work.graph
    assert len(space.edges_['work'][0]) == 6
    assert neighbor_pairs(space, 4, 'work') == list(
        zip(*[item.tolist() for item in space.edges_['work']], strict=True),
    )
    assert space._index_table(Adjacency(layer='work')).shape == (4, 3)
    assert space.run(0, Adjacency(layer='work')).tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    ('layers', 'message'),
    [
        ([Layer('work', nx.path_graph(5))], 'other nodes'),
        ([Layer('work', nx.path_graph(4)), Layer('work', nx.cycle_graph(4))], 'more than once'),
    ],
)
def test_network_space_rejects_inconsistent_layers(layers, message):
    """Test that a layer over other nodes and two layers of one label raise an error."""
    space = NetworkSpace(nx.path_graph(4), layers=layers)
    with pytest.raises(ValidationError, match=message):
        space.prepare()


def test_edges_of_an_unknown_layer_list_the_layers_of_the_space():
    """Test that counting the edges of an unknown layer raises an error that lists the layers of the space."""
    space = NetworkSpace(nx.path_graph(3), layers=[Layer('trade', nx.path_graph(3))])
    with pytest.raises(ValidationError, match="'trade'"):
        part_count(space, 'edges', 'ties', 'credit')


def test_network_space_runs_the_neighbours_of_a_node():
    """Test that running a prepared network space returns the neighbours of a node, and its facts are prepared."""
    space = NetworkSpace(nx.star_graph(3)).prepare()
    assert (space.size_, space.n_dim_, space.shape_, space.periodic_) == (4, 1, (4,), False)
    assert space.run(0).tolist() == [1, 2, 3]
    assert space.run(2).tolist() == [0]
    assert space.run(2, Adjacency(center=True)).tolist() == [2, 0]


@pytest.mark.parametrize('backend', BACKENDS)
def test_reductions_read_the_edges_of_their_layer(backend):
    """Test that reductions over two layers of one network sum the neighbours of each layer only."""
    contact, work = nx.path_graph(4), nx.Graph([(0, 3), (1, 2)])
    work.add_nodes_from(range(4))
    space = NetworkSpace(contact, layers=[Layer('work', work)])
    LayeredElement = Element.define(
        'LayeredElement',
        states={'y': 0.0, 'contacts': 0.0, 'colleagues': 0.0},
        laws=lambda s: [
            UpdateLaw(s.contacts, neighbors(s.other.y)),
            UpdateLaw(s.colleagues, neighbors(s.other.y, over=space.layers[0])),
        ],
    )
    values = np.array([1.0, 10.0, 100.0, 1000.0])
    system = System(components=[LayeredElement.lattice('g', space, y=values)])
    problem = compile_system(system, backend=backend)
    after = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(after, 'g.contacts').tolist() == [10.0, 101.0, 1010.0, 100.0]
    assert problem.states.read(after, 'g.colleagues').tolist() == [1000.0, 100.0, 10.0, 1.0]
