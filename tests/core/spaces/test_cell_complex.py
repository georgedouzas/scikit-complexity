"""Test the `_cell_complex` module of `skcomplex.core.spaces`."""

import itertools

import networkx as nx
import numpy as np
import pytest
import toponetx
import xgi

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.spaces import CellComplexSpace, LatticeSpace, NetworkSpace
from skcomplex.errors import SpaceError, ValidationError


def _is_zero(product):
    """Return whether a sparse product has no entry other than zero."""
    return product.count_nonzero() == 0


def _sizes(space):
    """Return the number of cells of every degree of a prepared complex."""
    return [space.incidences_[0].shape[1], *(matrix.shape[0] for matrix in space.incidences_)]


def _aligned(matrix, rows, columns):
    """Return a matrix with its rows and columns moved to the positions given by the matching of the cells."""
    aligned = np.zeros((len(rows), len(columns)))
    aligned[np.ix_(rows, columns)] = matrix
    return aligned


def _reference_signs(cells):
    """Return the sign of the order TopoNetX lists every cell in against its sorted order."""
    return np.array(
        [(-1.0) ** sum(a > b for a, b in itertools.combinations(tuple(cell), 2)) for cell in cells],
    )


def test_periodic_square_lattice_has_every_cell_and_vanishing_products():
    """Test the numbers of sites, links, and plaquettes of a periodic 8 by 8 lattice and its zero product."""
    space = CellComplexSpace(LatticeSpace((8, 8))).prepare()
    assert _sizes(space) == [64, 128, 64]
    assert _is_zero(space.incidences_[1] @ space.incidences_[0])


def test_periodic_links_point_along_the_positive_axis():
    """Test that the links that wrap around a periodic lattice run from the last site to the first."""
    space = CellComplexSpace(LatticeSpace((3, 3))).prepare()
    sources, targets = space.edges_[None]
    assert (sources[6], targets[6]) == (6, 0)
    assert space.hyperedges_[0] == (0, 3, 4, 1)


def test_periodic_lattice_of_two_sites_keeps_its_parallel_links():
    """Test that a lattice with two sites along an axis keeps both links between them."""
    space = CellComplexSpace(LatticeSpace((2, 2))).prepare()
    assert _sizes(space) == [4, 8, 4]
    assert _is_zero(space.incidences_[1] @ space.incidences_[0])
    assert space.graph.number_of_edges() == 8


@pytest.mark.parametrize('boundary', ['fixed', 'reflecting'])
def test_closed_edges_drop_the_cells_that_leave_the_box(boundary):
    """Test the numbers of cells of a 4 by 4 lattice with edges that close it, with no cell outside the box."""
    space = CellComplexSpace(LatticeSpace((4, 4), boundary=boundary)).prepare()
    assert _sizes(space) == [16, 24, 9]
    sources, targets = space.edges_[None]
    assert np.all(targets > sources)
    assert _is_zero(space.incidences_[1] @ space.incidences_[0])


def test_three_dimensional_lattice_holds_cubes():
    """Test that a periodic 3 by 3 by 3 lattice of dimension 3 holds its cubes, with zero consecutive products."""
    space = CellComplexSpace(LatticeSpace((3, 3, 3)), dimension=3).prepare()
    assert _sizes(space) == [27, 81, 81, 27]
    assert _is_zero(space.incidences_[1] @ space.incidences_[0])
    assert _is_zero(space.incidences_[2] @ space.incidences_[1])


def test_fixed_cubes_stay_in_the_box():
    """Test that a fixed 3 by 3 by 3 lattice holds only the cubes inside it."""
    space = CellComplexSpace(LatticeSpace((3, 3, 3), boundary='fixed'), dimension=3).prepare()
    assert _sizes(space) == [27, 54, 36, 8]
    assert _is_zero(space.incidences_[2] @ space.incidences_[1])


def test_lattice_stars_are_ratios_of_dual_to_primal_volumes():
    """Test the Hodge star of a lattice with different spacings along its axes."""
    space = CellComplexSpace(LatticeSpace((3, 3), spacing=(2.0, 0.5))).prepare()
    assert space.stars_[0][0] == 1.0
    assert (space.stars_[1][0], space.stars_[1][-1]) == (0.25, 4.0)
    assert space.stars_[2][0] == 1.0


def test_faces_of_a_network_come_from_a_minimum_cycle_basis():
    """Test that a 4-node cycle graph with a chord has 5 edges and 2 faces, each oriented by its node cycle."""
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    space = CellComplexSpace(NetworkSpace(graph)).prepare()
    assert _sizes(space) == [4, 5, 2]
    assert space.hyperedges_ == ((0, 1, 2), (0, 2, 3))
    assert _is_zero(space.incidences_[1] @ space.incidences_[0])
    sources, targets = space.edges_[None]
    edges = list(zip(sources.tolist(), targets.tolist(), strict=True))
    first = space.incidences_[1][[0], :].toarray().ravel()
    assert first[edges.index((0, 1))] == 1.0
    assert first[edges.index((1, 2))] == 1.0
    assert first[edges.index((0, 2))] == -1.0


def test_given_node_cycles_are_the_faces():
    """Test that a node cycle longer than a triangle is a face oriented by its given order."""
    space = CellComplexSpace(NetworkSpace(nx.cycle_graph(5)), faces=[(0, 4, 3, 2, 1)]).prepare()
    assert _sizes(space) == [5, 5, 1]
    assert _is_zero(space.incidences_[1] @ space.incidences_[0])
    assert space.incidences_[1].toarray()[0].tolist().count(-1.0) == 4


def test_clique_complex_of_the_complete_graph_holds_a_tetrahedron():
    """Test the cells of the clique complex of the complete graph on four nodes up to dimension 3."""
    space = CellComplexSpace(NetworkSpace(nx.complete_graph(4)), faces='cliques', dimension=3).prepare()
    assert _sizes(space) == [4, 6, 4, 1]
    assert _is_zero(space.incidences_[1] @ space.incidences_[0])
    assert _is_zero(space.incidences_[2] @ space.incidences_[1])


def test_simplicial_hyperedges_follow_the_boundary_matrix_of_xgi():
    """Test that a reversed triangle flips its column of the boundary matrix of XGI, with zero products."""
    graph = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)])
    network = NetworkSpace(graph, hyperedges=[(0, 1, 2), (3, 2, 1)])
    space = CellComplexSpace(network, faces='hyperedges').prepare()
    assert _is_zero(space.incidences_[1] @ space.incidences_[0])
    sources, targets = space.edges_[None]
    edges = [frozenset(item) for item in zip(sources.tolist(), targets.tolist(), strict=True)]
    simplices = xgi.SimplicialComplex()
    simplices.add_simplices_from([[0, 1, 2], [1, 2, 3]])
    boundary, rows, columns = xgi.linalg.hodge_matrix.boundary_matrix(simplices, 2, index=True)
    expected = np.zeros((len(edges), 2))
    for row, column in itertools.product(range(boundary.shape[0]), range(boundary.shape[1])):
        members = frozenset(simplices.edges.members(columns[column]))
        place = 0 if members == frozenset({0, 1, 2}) else 1
        flip = -1.0 if place == 1 else 1.0
        expected[edges.index(frozenset(simplices.edges.members(rows[row]))), place] = flip * boundary[row, column]
    assert np.array_equal(space.incidences_[1].T.toarray(), expected)


def test_cycle_tables_agree_with_the_boundary_matrices_of_xgi():
    """Test that faces given as cycles, which the tables sum, give the incidence that XGI gives their cliques."""
    graph = nx.gnp_random_graph(9, 0.5, seed=3)
    cliques = CellComplexSpace(NetworkSpace(graph), faces='cliques').prepare()
    cycles = CellComplexSpace(NetworkSpace(graph), faces=list(cliques.hyperedges_)).prepare()
    assert (cliques.incidences_[1] != cycles.incidences_[1]).nnz == 0
    assert (cliques.incidences_[0] != cycles.incidences_[0]).nnz == 0


def test_incidence_parity_with_xgi_and_toponetx():
    """Test the clique complex of a random graph on 12 nodes entry by entry against XGI and TopoNetX."""
    graph = nx.gnp_random_graph(12, 0.6, seed=7)
    space = CellComplexSpace(NetworkSpace(graph), faces='cliques').prepare()
    sources, targets = space.edges_[None]
    edges = list(zip(sources.tolist(), targets.tolist(), strict=True))
    triangles = list(space.hyperedges_)
    numbering = [
        {frozenset(cell): number for number, cell in enumerate(cells)}
        for cells in ([(node,) for node in range(12)], edges, triangles)
    ]
    simplices = xgi.SimplicialComplex()
    simplices.add_nodes_from(range(12))
    simplices.add_simplices_from([list(item) for item in [*edges, *triangles]])
    complex_ = toponetx.SimplicialComplex([list(item) for item in [*edges, *triangles]])
    for order in (1, 2):
        ours = space.incidences_[order - 1].T.toarray()
        boundary, rows, columns = xgi.linalg.hodge_matrix.boundary_matrix(simplices, order, index=True)
        lows = [
            numbering[order - 1][frozenset([rows[row]] if order == 1 else simplices.edges.members(rows[row]))]
            for row in range(boundary.shape[0])
        ]
        highs = [
            numbering[order][frozenset(simplices.edges.members(columns[column]))] for column in range(boundary.shape[1])
        ]
        assert np.array_equal(ours, _aligned(boundary, lows, highs))
        lower, upper, matrix = complex_.incidence_matrix(order, index=True)
        lows = [numbering[order - 1][frozenset(item)] for item in lower]
        highs = [numbering[order][frozenset(item)] for item in upper]
        signs = _reference_signs(lower)[:, None] * _reference_signs(upper)
        assert np.array_equal(ours[np.ix_(lows, highs)], signs * matrix.toarray())


def test_invalid_face_names_the_face_and_the_missing_edge():
    """Test that a face over a pair the graph does not join raises an error naming both."""
    space = CellComplexSpace(NetworkSpace(nx.cycle_graph(5)), faces=[(0, 1, 3)])
    with pytest.raises(SpaceError, match=r'\(0, 1, 3\).*\(1, 3\)'):
        space.prepare()


def test_invalid_hyperedge_raises_a_space_error():
    """Test that a hyperedge over a pair the graph does not join raises an error."""
    network = NetworkSpace(nx.path_graph(3), hyperedges=[(0, 1, 2)])
    with pytest.raises(SpaceError, match=r'\(2, 0\)'):
        CellComplexSpace(network, faces='hyperedges').prepare()


def test_face_of_two_nodes_raises_a_space_error():
    """Test that a face needs at least three nodes."""
    with pytest.raises(SpaceError, match='at least 3'):
        CellComplexSpace(NetworkSpace(nx.path_graph(3)), faces=[(0, 1)]).prepare()


def test_loop_of_a_network_raises_a_space_error():
    """Test that a network with a loop has no cell complex."""
    graph = nx.path_graph(3)
    graph.add_edge(1, 1)
    with pytest.raises(SpaceError, match='loops'):
        CellComplexSpace(NetworkSpace(graph)).prepare()


@pytest.mark.parametrize(
    ('arguments', 'error'),
    [
        ({'space': 'nowhere'}, SpaceError),
        ({'space': LatticeSpace((3, 3), geometry='triangular')}, SpaceError),
        ({'space': LatticeSpace((3, 3)), 'faces': [(0, 1, 2)]}, SpaceError),
        ({'space': LatticeSpace((3, 3)), 'dimension': 1}, ValidationError),
        ({'space': NetworkSpace(nx.cycle_graph(3)), 'faces': 'simplices'}, ValidationError),
    ],
)
def test_check_rejects_arguments_the_complex_cannot_take(arguments, error):
    """Test that a space, faces, or a dimension the complex cannot take raise an error."""
    with pytest.raises(error):
        CellComplexSpace(**arguments).prepare()


def test_cofaces_list_the_faces_of_every_link_with_their_signs():
    """Test the padded table of the faces of the links of a 3 by 3 fixed lattice."""
    space = CellComplexSpace(LatticeSpace((3, 3), boundary='fixed')).prepare()
    table, signs = space.cofaces_[1], space.orientations_[1]
    assert table.shape == (12, 2)
    assert np.sum(table == 4) == 8
    for link, slot in itertools.product(range(12), range(2)):
        if table[link, slot] < 4:
            assert space.incidences_[1][int(table[link, slot]), link] == signs[link, slot]


def test_potential_and_field_strength_groups_hold_one_member_per_cell():
    """Test that groups on the edges and hyperedges of a periodic 16 by 16 complex hold 512 and 256 values."""
    space = CellComplexSpace(LatticeSpace((16, 16)))
    potentials = GroupElement('potentials', element=Element, space=space, on='edges').prepare()
    strengths = GroupElement('strengths', element=Element, space=space, on='hyperedges').prepare()
    assert (potentials.size_, strengths.size_) == (512, 256)
