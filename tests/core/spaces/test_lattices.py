"""Test lattice spaces: their checks, their prepared facts, and the neighbours of their sites."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.spaces import BoxNeighborhood, LatticeSpace


def test_lattice_space_prepares_its_facts():
    """Test that a prepared lattice of shape (4, 5) has 20 sites in two dimensions and 20 positions."""
    space = LatticeSpace((4, 5), spacing=0.5).prepare()
    assert (space.size_, space.n_dim_, space.shape_) == (20, 2, (4, 5))
    assert space.positions_.shape == (20, 2)
    assert space.positions_[-1].tolist() == [1.5, 2.0]
    assert (space.spacings_, space.periodic_, space.default_visual_) == ((0.5, 0.5), True, 'image')


def test_lattice_space_with_a_negative_side_is_checked_not_built():
    """Test that a lattice with a negative side is built without error and rejected by its check."""
    space = LatticeSpace((-1, 4))
    with pytest.raises(ValidationError, match='positive number of sites'):
        space.prepare()


def test_lattice_space_runs_the_neighbours_of_its_own_neighbourhood():
    """Test that a site reads the von Neumann or the Moore neighbourhood of the space, or the one it is given."""
    assert LatticeSpace(5, boundary='fixed', neighborhood='von_neumann').prepare().run(0).tolist() == [1]
    assert len(LatticeSpace((3, 3)).prepare().run(4)) == 8
    assert LatticeSpace(5).prepare().run(0, BoxNeighborhood()).tolist() == [4, 0, 1]


def test_lattice_space_balls_and_neighbour_pairs():
    """Test that the ball of a lattice is the box of offsets and that its pairs join sites one step apart."""
    space = LatticeSpace(3, boundary='fixed').prepare()
    assert space._ball(1).offsets == ((-1,), (0,), (1,))
    assert space.prepare()._neighbor_pairs() == [(0, 1), (1, 2)]
