"""Test network spaces: generators, layers, and the arrays of their edges and adjacency."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.base import ValidationError
from skcomplex.core.compile import compile_system
from skcomplex.core.components import Element
from skcomplex.core.laws import Update, neighbors
from skcomplex.core.space import Adjacency, Layer, NetworkSpace, neighbour_pairs
from skcomplex.core.system import System

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
    sources, targets = space.list_edges()
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
    assert first.size == second.size
    assert _edges(first) == _edges(second)


def test_deterministic_generators_build_their_graphs():
    """Test that the complete and grid networks have the edges of their graphs and the grid its coordinates."""
    assert NetworkSpace.complete(6).graph.number_of_edges() == 15
    grid = NetworkSpace.grid((3, 4))
    assert grid.size == 12
    assert grid.read_positions()[5].tolist() == [1.0, 1.0]
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
    space = NetworkSpace.random_geometric(60, 0.2, seed=2)
    positions = space.read_positions()
    sources, targets = space.list_edges()
    assert positions.shape == (60, 2)
    assert np.all(np.linalg.norm(positions[sources] - positions[targets], axis=1) <= 0.2)
    assert NetworkSpace.waxman(30, seed=1).read_positions().shape == (30, 2)


def test_read_positions_rejects_nodes_without_a_position():
    """Test that reading positions of a graph without them names the missing attribute."""
    with pytest.raises(ValidationError, match="'pos'"):
        NetworkSpace(nx.path_graph(3)).read_positions()


def test_list_edges_orders_undirected_ends_and_keeps_directions():
    """Test that the source of an undirected edge comes first in node order and a directed edge keeps its direction."""
    undirected = nx.Graph()
    undirected.add_nodes_from(['a', 'b', 'c'])
    undirected.add_edges_from([('c', 'a'), ('b', 'c')])
    assert [item.tolist() for item in NetworkSpace(undirected).list_edges()] == [[0, 1], [2, 2]]
    directed = NetworkSpace(nx.DiGraph([(2, 0), (0, 1)]))
    assert list(directed.graph.nodes) == [2, 0, 1]
    assert [item.tolist() for item in directed.list_edges()] == [[0, 1], [1, 2]]


def test_compute_adjacency_lists_neighbours_with_their_edges():
    """Test that the adjacency lists every neighbour with the edge that joins it, successors on a directed graph."""
    space = NetworkSpace(nx.Graph([(0, 1), (1, 2), (0, 2)]))
    indptr, indices, edges = space.compute_adjacency()
    sources, targets = space.list_edges()
    for node in range(3):
        for neighbour, edge in zip(
            indices[indptr[node] : indptr[node + 1]],
            edges[indptr[node] : indptr[node + 1]],
            strict=True,
        ):
            assert {int(sources[edge]), int(targets[edge])} == {node, int(neighbour)}
    directed = NetworkSpace(nx.DiGraph([(0, 1), (2, 1)]))
    indptr, indices, _ = directed.compute_adjacency()
    assert np.diff(indptr).tolist() == [1, 0, 1]


def test_layers_hold_relations_over_the_same_nodes():
    """Test that a network space resolves the graph and the edges of every layer by its label."""
    work = Layer('work', nx.complete_graph(4))
    space = NetworkSpace(nx.path_graph(4), layers=[work])
    assert space.layer_labels == (None, 'work')
    assert space.resolve_graph('work') is work.graph
    assert len(space.list_edges('work')[0]) == 6
    assert neighbour_pairs(space, 4, 'work') == list(
        zip(*[item.tolist() for item in space.list_edges('work')], strict=True),
    )
    assert space.index_table(Adjacency(layer='work')).shape == (4, 3)


@pytest.mark.parametrize(
    ('layers', 'message'),
    [
        ([Layer('work', nx.path_graph(5))], 'other nodes'),
        ([Layer('work', nx.path_graph(4)), Layer('work', nx.cycle_graph(4))], 'more than once'),
    ],
)
def test_network_space_rejects_inconsistent_layers(layers, message):
    """Test that a layer over other nodes and two layers of one label raise an error."""
    with pytest.raises(ValidationError, match=message):
        NetworkSpace(nx.path_graph(4), layers=layers)


def test_resolve_graph_lists_the_layers_of_the_space():
    """Test that an unknown layer raises an error that lists the layers of the space."""
    space = NetworkSpace(nx.path_graph(3), layers=[Layer('trade', nx.path_graph(3))])
    with pytest.raises(ValidationError, match="'trade'"):
        space.resolve_graph('credit')


@pytest.mark.parametrize('backend', BACKENDS)
def test_reductions_read_the_edges_of_their_layer(backend):
    """Test that reductions over two layers of one network sum the neighbours of each layer only."""
    contact, work = nx.path_graph(4), nx.Graph([(0, 3), (1, 2)])
    work.add_nodes_from(range(4))
    space = NetworkSpace(contact, layers=[Layer('work', work)])
    Layered = Element.define(
        'Layered',
        states={'y': 0.0, 'contacts': 0.0, 'colleagues': 0.0},
        laws=lambda s: [
            Update(s.contacts, neighbors(s.other.y)),
            Update(s.colleagues, neighbors(s.other.y, over=space.layers[0])),
        ],
    )
    values = np.array([1.0, 10.0, 100.0, 1000.0])
    system = System(components=[Layered.lattice('g', space, y=values)])
    problem = compile_system(system, backend=backend)
    after = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(after, 'g.contacts').tolist() == [10.0, 101.0, 1010.0, 100.0]
    assert problem.states.read(after, 'g.colleagues').tolist() == [1000.0, 100.0, 10.0, 1.0]
