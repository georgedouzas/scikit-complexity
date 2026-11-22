"""Test the `_lattice` module of `skcomplex.core.components.spaces`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import (
    AdjacencyNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    NetworkSpace,
)
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System


def test_lattice_space_prepares_its_facts():
    """Test that a prepared lattice of shape (4, 5) has 20 sites in two dimensions and 20 positions."""
    space = LatticeSpace((4, 5), spacing=0.5).prepare()
    assert (space.size_, space.n_dim_, space.shape_) == (20, 2, (4, 5))
    assert space.positions_.shape == (20, 2)
    assert space.positions_[-1].tolist() == [1.5, 2.0]
    assert (space.spacings_, space.periodic_, space.default_visual_) == ((0.5, 0.5), True, 'image')


@pytest.mark.parametrize(
    ('shape', 'axes'),
    [(10, ('x',)), ((4, 5), ('x', 'y')), ((2, 2, 2, 2), ('x0', 'x1', 'x2', 'x3'))],
)
def test_lattice_space_names_the_coordinates_of_its_cells(shape, axes):
    """Test that the coordinates of a cell are named `x`, `y`, and `z`, or `x0`, `x1`, and so on beyond three."""
    assert LatticeSpace(shape).prepare().axes_ == axes


@pytest.mark.parametrize(
    ('space', 'message'),
    [
        (LatticeSpace((-1, 4)), 'positive number of sites'),
        (LatticeSpace(4, boundary='open'), 'Unknown lattice boundary'),
        (LatticeSpace(4, geometry='hexagonal'), 'Unknown lattice geometry'),
        (LatticeSpace((2, 2, 2), geometry='triangular'), 'two dimensions'),
        (LatticeSpace(4, metric='hexagonal'), 'square lattice has the metric'),
        (LatticeSpace((3, 3), geometry='triangular', metric='manhattan'), 'triangular lattice has the metric'),
        (LatticeSpace(4, metric='euclidean'), 'Use one of chebyshev, manhattan'),
    ],
)
def test_invalid_lattice_space_is_checked_not_built(space, message):
    """Test that an invalid lattice is built without error and rejected by its check."""
    with pytest.raises(ValidationError, match=message):
        space.prepare()


def test_lattice_space_reads_the_neighbours_of_its_own_neighbourhood():
    """Test that a site reads the ball of radius 1 under the metric of the space, or the neighbourhood it is given."""
    fixed = LatticeSpace(5, boundary='fixed', metric='manhattan').prepare()
    triangular = LatticeSpace((3, 3), geometry='triangular').prepare()
    assert fixed._get_neighbors(0).tolist() == [1]
    assert len(LatticeSpace((3, 3)).prepare()._get_neighbors(4)) == 8
    box = LatticeNeighborhood(((-1,), (0,), (1,)))
    assert LatticeSpace(5).prepare()._get_neighbors(0, box).tolist() == [4, 0, 1]
    assert triangular._get_default_neighborhood()._get_offsets(2) == (
        (0, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
    )
    assert (triangular.metric_, fixed.metric_, LatticeSpace(5).prepare().metric_) == (
        'hexagonal',
        'manhattan',
        'chebyshev',
    )


def test_lattice_space_ball_keeps_the_site_inline():
    """Test that the ball an analysis reads is the Chebyshev ball with the site in its lexicographic place."""
    ball = LatticeSpace((3, 3), geometry='triangular').prepare()._get_ball(1).prepare()
    assert ball.offsets_ == ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


def test_lattice_space_reflects_at_reflecting_edges():
    """Test that a reflecting lattice reads the mirror image of the box outside it."""
    space = LatticeSpace(4, boundary='reflecting').prepare()
    assert space._get_index_table(LatticeNeighborhood(((-1,), (0,), (1,))))[:, 0].tolist() == [0, 0, 1, 2]
    assert space._get_index_table(LatticeNeighborhood(((-1,), (0,), (1,))))[:, 2].tolist() == [1, 2, 3, 3]


def test_lattice_space_rejects_neighbourhoods_it_cannot_read():
    """Test that a lattice rejects an adjacency neighbourhood and offsets of another dimension."""
    space = LatticeSpace((3, 3)).prepare()
    with pytest.raises(ValidationError, match='lattice neighbourhood'):
        space._get_index_table(AdjacencyNeighborhood())
    with pytest.raises(ValidationError, match='dimension'):
        space._get_index_table(LatticeNeighborhood(((-1,), (0,), (1,))))


def test_lattice_space_balls_and_neighbour_pairs():
    """Test that the ball of a lattice is the box of offsets and that its pairs join sites one step apart."""
    space = LatticeSpace(3, boundary='fixed').prepare()
    assert space._get_ball(1).prepare().offsets_ == ((-1,), (0,), (1,))
    assert space._get_neighbor_pairs() == [(0, 1), (1, 2)]
    assert sorted(LatticeSpace(3).prepare()._get_neighbor_pairs()) == [(0, 1), (1, 2), (2, 0)]


def test_neighbor_pairs_list_lattice_and_network_bonds():
    """Test that neighbour pairs are the bonds of periodic and open lattices and of networks."""
    assert LatticeSpace(4).prepare()._get_neighbor_pairs() == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert LatticeSpace(3, boundary='fixed').prepare()._get_neighbor_pairs() == [(0, 1), (1, 2)]
    assert len(LatticeSpace((2, 3)).prepare()._get_neighbor_pairs()) == 9
    assert NetworkSpace(nx.path_graph(3)).prepare()._get_neighbor_pairs() == [(0, 1), (1, 2)]


@pytest.mark.parametrize(
    ('space', 'message'),
    [
        (LatticeSpace(4, discretization='finite-element'), 'Unknown discretization'),
        (LatticeSpace(4, boundary=['u']), 'Unknown lattice boundary'),
    ],
)
def test_lattice_space_checks_how_a_field_on_it_becomes_numbers(space, message):
    """Test that a lattice rejects a discretization it has no operators for and a boundary of the wrong kind."""
    with pytest.raises(ValidationError, match=message):
        space.prepare()


def test_lattice_space_with_the_edge_of_each_state_joins_its_members_periodically():
    """Test that a lattice whose boundary names the edge of a state joins the neighbours of its members periodically."""
    space = LatticeSpace(5, boundary={'u': ('fixed', 1.0)}, discretization='spectral').prepare()
    box = LatticeNeighborhood(((-1,), (0,), (1,)))
    assert space.periodic_
    assert space._get_neighbors(0, box).tolist() == [4, 0, 1]


def _heat(s):
    """Return the diffusion of a field `u`."""
    return [RateLaw(s.target.u.diff(s.t), laplacian(s.target.u))]


def _heated(space):
    """Return a prepared system of a field `u`, zero at first, that diffuses on a space."""
    heat = Interaction('heat', roles={'target': 'f'}, laws=_heat)
    return System(elements=[Element('f', values={'u': 0.0}, space=space)], interactions=[heat]).prepare()


@pytest.mark.parametrize(
    ('space', 'edge'),
    [(LatticeSpace(4, boundary='fixed', fill=2.0), 2.0), (LatticeSpace(4), 0.0)],
)
def test_field_on_a_lattice_takes_the_edges_of_the_lattice(space, edge):
    """Test that a field without edges of its own takes those of its lattice, such as fixed edges of a fill value."""
    final = _heated(space).simulate(until=50.0, dt=25.0).get_values('f.u')[-1]
    np.testing.assert_allclose(final, edge, atol=1e-5)


def test_field_on_a_lattice_rejects_an_unknown_edge_of_its_state():
    """Test that an unknown edge of one state raises an error that names the state and the known edges."""
    with pytest.raises(ValidationError, match=r"Unknown boundary 'open' of f\.u"):
        _heated(LatticeSpace(4, boundary={'u': 'open'}))
