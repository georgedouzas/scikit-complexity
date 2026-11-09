"""Test the `_lattice` module of `skcomplex.core.components.spaces`."""

import pytest

from skcomplex.core.components.spaces import AdjacencyNeighborhood, BallNeighborhood, LatticeSpace
from skcomplex.core.errors import ValidationError


def test_lattice_space_prepares_its_facts():
    """Test that a prepared lattice of shape (4, 5) has 20 sites in two dimensions and 20 positions."""
    space = LatticeSpace((4, 5), spacing=0.5).prepare()
    assert (space.size_, space.n_dim_, space.shape_) == (20, 2, (4, 5))
    assert space.positions_.shape == (20, 2)
    assert space.positions_[-1].tolist() == [1.5, 2.0]
    assert (space.spacings_, space.periodic_, space.default_visual_) == ((0.5, 0.5), True, 'image')


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
    box = BallNeighborhood(n_dim=1, center='inline')
    assert LatticeSpace(5).prepare()._get_neighbors(0, box).tolist() == [4, 0, 1]
    assert triangular._get_default_neighborhood().prepare().offsets_ == (
        BallNeighborhood(metric='hexagonal').prepare().offsets_
    )
    assert (triangular.metric_, fixed.metric_, LatticeSpace(5).prepare().metric_) == (
        'hexagonal',
        'manhattan',
        'chebyshev',
    )


def test_lattice_space_ball_keeps_the_site_inline():
    """Test that the ball an analysis reads is the Chebyshev ball with the site in its lexicographic place."""
    ball = LatticeSpace((3, 3), geometry='triangular').prepare()._get_ball(1).prepare()
    assert ball.offsets_ == BallNeighborhood(center='inline').prepare().offsets_


def test_lattice_space_reflects_at_reflecting_edges():
    """Test that a reflecting lattice reads the mirror image of the box outside it."""
    space = LatticeSpace(4, boundary='reflecting').prepare()
    assert space._get_index_table(BallNeighborhood(n_dim=1, center='inline'))[:, 0].tolist() == [0, 0, 1, 2]
    assert space._get_index_table(BallNeighborhood(n_dim=1, center='inline'))[:, 2].tolist() == [1, 2, 3, 3]


def test_lattice_space_rejects_neighbourhoods_it_cannot_read():
    """Test that a lattice rejects an adjacency neighbourhood and offsets of another dimension."""
    space = LatticeSpace((3, 3)).prepare()
    with pytest.raises(ValidationError, match='lattice neighbourhood'):
        space._get_index_table(AdjacencyNeighborhood())
    with pytest.raises(ValidationError, match='dimension'):
        space._get_index_table(BallNeighborhood(n_dim=1, center='inline'))


def test_lattice_space_balls_and_neighbour_pairs():
    """Test that the ball of a lattice is the box of offsets and that its pairs join sites one step apart."""
    space = LatticeSpace(3, boundary='fixed').prepare()
    assert space._get_ball(1).prepare().offsets_ == ((-1,), (0,), (1,))
    assert space._get_neighbor_pairs() == [(0, 1), (1, 2)]
    assert sorted(LatticeSpace(3).prepare()._get_neighbor_pairs()) == [(0, 1), (1, 2), (2, 0)]
