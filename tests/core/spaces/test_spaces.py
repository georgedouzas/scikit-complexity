"""Test the continuous space: its arguments, its checks, and its prepared facts."""

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import ValidationError
from skcomplex.core.spaces import ContinuousSpace


def test_continuous_space_constructor_only_stores_its_arguments():
    """Test that an invalid box is built without error and rejected by its check."""
    space = ContinuousSpace(extent=(1.0, -1.0))
    assert space.sides == (1.0, -1.0)
    with pytest.raises(ValidationError, match='positive side'):
        space.check()


def test_continuous_space_prepares_its_box():
    """Test that a prepared box has its extent, its dimension, its boundary, and no sites."""
    space = ContinuousSpace(extent=(10.0, 5.0), boundary='periodic').prepare()
    assert space.extent_ == (10.0, 5.0)
    assert (space.size_, space.n_dim_, space.shape_, space.periodic_) == (None, 2, (), True)
    assert space.positions_.shape == (0, 2)
    assert space.default_visual_ is None


def test_unbounded_continuous_space_takes_its_dimension():
    """Test that an unbounded space has the given dimension and no extent."""
    space = ContinuousSpace(n_dim=3).prepare()
    assert (space.extent_, space.n_dim_, space.periodic_) == ((), 3, False)


def test_continuous_space_has_no_neighbours_of_a_site():
    """Test that a site of a continuous space has no neighbours, since positions are states of the members."""
    assert ContinuousSpace(extent=(1.0, 1.0)).prepare().run(0).tolist() == []


def test_continuous_space_clones_from_its_sides():
    """Test that a clone of a box given by its sides has the same extent and boundary."""
    space = clone(ContinuousSpace(extent=(2.0, 3.0), boundary='reflecting'))
    assert space.prepare().extent_ == (2.0, 3.0)
    assert space.boundary == 'reflecting'
    assert np.array_equal(space.positions_, np.zeros((0, 2)))
