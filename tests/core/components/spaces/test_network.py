"""Test the `_network` module of `skcomplex.core.components.spaces`."""

import itertools
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import xgi

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import AdjacencyNeighborhood, NetworkSpace, VonNeumannNeighborhood
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import SpaceError, ValidationError
from skcomplex.core.specs import LayerSpec, StateSpec, VisualSpec

BACKENDS = (NumpyBackend(), NumbaBackend())
GENERATORS = [
    lambda seed: NetworkSpace(nx.gnp_random_graph(40, 0.1, seed=seed)),
    lambda seed: NetworkSpace(nx.watts_strogatz_graph(40, 4, 0.2, seed=seed)),
    lambda seed: NetworkSpace(nx.barabasi_albert_graph(40, 2, seed=seed)),
    lambda seed: NetworkSpace(nx.Graph(nx.configuration_model([3] * 20 + [1] * 20, seed=seed))),
    lambda seed: NetworkSpace(nx.stochastic_block_model([20, 20], [[0.3, 0.02], [0.02, 0.3]], seed=seed)),
    lambda seed: NetworkSpace(nx.waxman_graph(40, seed=seed)),
    lambda seed: NetworkSpace(nx.random_geometric_graph(40, 0.25, seed=seed)),
]


def _edges(space):
    """Return the edges of a network space as a sorted list of node index pairs."""
    sources, targets = space.prepare().edges_[None]
    return sorted(zip(sources.tolist(), targets.tolist(), strict=True))


def test_watts_strogatz_with_the_same_seed_gives_the_same_network():
    """Test that a Watts-Strogatz network generated twice with the same seed has identical edges."""
    first = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.3, seed=7))
    second = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.3, seed=7))
    assert _edges(first) == _edges(second)
    assert _edges(first) != _edges(NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.3, seed=8)))


@pytest.mark.parametrize('generate', GENERATORS)
def test_generators_reproduce_their_network_from_a_random_stream(generate):
    """Test that every random generator reproduces its network from generators of the same random stream."""
    first = generate(np.random.default_rng(3))
    second = generate(np.random.default_rng(3))
    assert first.prepare().size_ == second.prepare().size_
    assert _edges(first) == _edges(second)


def test_spatial_generators_store_node_positions():
    """Test that the Waxman and random geometric networks store positions that join near nodes."""
    space = NetworkSpace(nx.random_geometric_graph(60, 0.2, seed=2)).prepare()
    positions = space.positions_
    sources, targets = space.edges_[None]
    assert positions.shape == (60, 2)
    assert np.all(np.linalg.norm(positions[sources] - positions[targets], axis=1) <= 0.2)
    assert NetworkSpace(nx.waxman_graph(30, seed=1)).prepare().positions_.shape == (30, 2)


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
    work = LayerSpec('work', nx.complete_graph(4))
    space = NetworkSpace(nx.path_graph(4), layers=[work]).prepare()
    assert list(space.graphs_) == [None, 'work']
    assert space.graphs_['work'] is work.graph
    assert len(space.edges_['work'][0]) == 6
    assert space.prepare()._neighbor_pairs('work') == list(
        zip(*[item.tolist() for item in space.edges_['work']], strict=True),
    )
    assert space._index_table(AdjacencyNeighborhood(layer='work')).shape == (4, 3)
    assert space._get_neighbors(0, AdjacencyNeighborhood(layer='work')).tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    ('layers', 'message'),
    [
        ([LayerSpec('work', nx.path_graph(5))], 'other nodes'),
        ([LayerSpec('work', nx.path_graph(4)), LayerSpec('work', nx.cycle_graph(4))], 'more than once'),
    ],
)
def test_network_space_rejects_inconsistent_layers(layers, message):
    """Test that a layer over other nodes and two layers of one label raise an error."""
    space = NetworkSpace(nx.path_graph(4), layers=layers)
    with pytest.raises(ValidationError, match=message):
        space.prepare()


def test_edges_of_an_unknown_layer_list_the_layers_of_the_space():
    """Test that counting the edges of an unknown layer raises an error that lists the layers of the space."""
    space = NetworkSpace(nx.path_graph(3), layers=[LayerSpec('trade', nx.path_graph(3))])
    with pytest.raises(SpaceError, match="'trade'"):
        space.prepare()._count_parts('edges', 'credit')


def test_network_space_counts_its_hyperedges():
    """Test that a network counts its hyperedges as parts and rejects a hyperedge outside its nodes."""
    space = NetworkSpace(nx.path_graph(4), hyperedges=[(0, 1, 2), (1, 3)]).prepare()
    assert space._count_parts('hyperedges') == 2
    with pytest.raises(ValidationError, match='at least two nodes'):
        NetworkSpace(nx.path_graph(4), hyperedges=[(0, 7)]).prepare()


def test_network_space_reads_an_adjacency_neighbourhood_only():
    """Test that a network rejects a lattice neighbourhood."""
    with pytest.raises(ValidationError, match='adjacency neighbourhood'):
        NetworkSpace(nx.path_graph(4)).prepare()._index_table(VonNeumannNeighborhood())


def test_network_space_gives_the_neighbours_of_a_node():
    """Test that a prepared network space has its facts and gives the neighbours of a node."""
    space = NetworkSpace(nx.star_graph(3)).prepare()
    assert (space.size_, space.n_dim_, space.shape_, space.periodic_) == (4, 1, (4,), False)
    assert space._get_neighbors(0).tolist() == [1, 2, 3]
    assert space._get_neighbors(2).tolist() == [0]
    assert space._get_neighbors(2, AdjacencyNeighborhood(center=True)).tolist() == [2, 0]


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_reductions_read_the_edges_of_their_layer(backend):
    """Test that reductions over two layers of one network sum the neighbours of each layer only."""
    contact, work = nx.path_graph(4), nx.Graph([(0, 3), (1, 2)])
    work.add_nodes_from(range(4))
    space = NetworkSpace(contact, layers=[LayerSpec('work', work)])

    class LayeredElement(Element):
        """Layered element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            y: Annotated[object, StateSpec()] = 0.0,
            contacts: Annotated[object, StateSpec()] = 0.0,
            colleagues: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.y = y
            self.contacts = contacts
            self.colleagues = colleagues

        def _get_laws(self, s):
            """Return the laws of the layered element."""
            return [
                UpdateLaw(s.contacts, neighbors(s.other.y)),
                UpdateLaw(s.colleagues, neighbors(s.other.y, over=space.layers[0])),
            ]

    values = np.array([1.0, 10.0, 100.0, 1000.0])
    system = System(elements=[LatticeElement('g', element=LayeredElement, space=space, values={'y': values})])
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    after = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(after, 'g.contacts').tolist() == [10.0, 101.0, 1010.0, 100.0]
    assert problem.states.read(after, 'g.colleagues').tolist() == [1000.0, 100.0, 10.0, 1.0]


def test_network_space_takes_the_triangles_of_an_xgi_hypergraph():
    """Test that the triangles of a complete graph on five nodes give 10 hyperedges of size 3, 6 around every node."""
    triangles = xgi.Hypergraph([list(nodes) for nodes in itertools.combinations(range(5), 3)])
    space = NetworkSpace(nx.complete_graph(5), hyperedges=triangles).prepare()
    assert len(space.hyperedges_) == 10
    assert {len(hyperedge) for hyperedge in space.hyperedges_} == {3}
    assert [len(memberships) for memberships in space.memberships_] == [6] * 5
    assert space.incidence_.shape == (5, 10)
    assert np.array_equal(space.incidence_.sum(axis=0), np.full(10, 3.0))
    assert space._count_parts('hyperedges') == 10


def test_network_space_takes_the_faces_of_an_xgi_simplicial_complex():
    """Test that a simplicial complex gives its faces of at least two nodes, members in the order of the graph."""
    complex_ = xgi.SimplicialComplex([['c', 'a', 'b'], ['b', 'd']])
    graph = nx.Graph([('a', 'b'), ('b', 'c'), ('a', 'c'), ('b', 'd')])
    space = NetworkSpace(graph, hyperedges=complex_).prepare()
    order = list(graph.nodes)
    named = sorted(tuple(order[index] for index in hyperedge) for hyperedge in space.hyperedges_)
    assert named == [('a', 'b'), ('a', 'b', 'c'), ('a', 'c'), ('b', 'c'), ('b', 'd')]
    assert all(list(hyperedge) == sorted(hyperedge) for hyperedge in space.hyperedges_)


def test_network_space_rejects_an_xgi_hyperedge_outside_its_nodes():
    """Test that an XGI hyperedge with a node outside the graph raises an error."""
    with pytest.raises(ValidationError, match='at least two nodes'):
        NetworkSpace(nx.path_graph(3), hyperedges=xgi.Hypergraph([[0, 1, 9]])).prepare()


def test_network_space_without_hyperedges_has_an_empty_incidence():
    """Test that a network without hyperedges has no hyperedges and an incidence matrix with no columns."""
    space = NetworkSpace(nx.path_graph(3)).prepare()
    assert space.hyperedges_ == ()
    assert space.incidence_.shape == (3, 0)
    assert space.memberships_ == ((), (), ())
