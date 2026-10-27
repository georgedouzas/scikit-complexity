"""Test the `_continuous` module of `skcomplex.core.spaces`."""

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.errors import ValidationError
from skcomplex.core.spaces import ContinuousSpace


@pytest.mark.parametrize(
    ('space', 'message'),
    [
        (ContinuousSpace(extent=(1.0, -1.0)), 'positive side'),
        (ContinuousSpace(extent=(1.0, 1.0), n_dim=3), 'positive side'),
        (ContinuousSpace(), 'number of dimensions'),
        (ContinuousSpace(extent=(1.0,), boundary='sticky'), 'Unknown boundary'),
        (ContinuousSpace(n_dim=2, boundary='periodic'), 'no walls'),
    ],
)
def test_continuous_space_constructor_only_stores_its_arguments(space, message):
    """Test that an invalid space is built without error and rejected by its check."""
    with pytest.raises(ValidationError, match=message):
        space.check()


def test_continuous_space_prepares_its_box():
    """Test that a prepared box has its extent, its dimension, its boundary, and no sites."""
    space = ContinuousSpace(extent=(10.0, 5.0), boundary='periodic').check().prepare()
    assert space.extent_ == (10.0, 5.0)
    assert (space.size_, space.n_dim_, space.shape_, space.periodic_) == (None, 2, (), True)
    assert space.positions_.shape == (0, 2)
    assert space.default_visual_ is None


def test_unbounded_continuous_space_takes_its_dimension():
    """Test that an unbounded space has the given dimension and no extent."""
    space = ContinuousSpace(n_dim=3).check().prepare()
    assert (space.extent_, space.n_dim_, space.periodic_) == ((), 3, False)


def test_continuous_space_has_no_neighbours_of_a_site():
    """Test that a site of a continuous space has no neighbours, since positions are states of the members."""
    assert ContinuousSpace(extent=(1.0, 1.0)).check().prepare()._get_neighbors(0).tolist() == []


def test_continuous_space_clones_from_its_arguments():
    """Test that a clone of a box has the same extent and boundary."""
    space = clone(ContinuousSpace(extent=(2.0, 3.0), boundary='reflecting')).check().prepare()
    assert space.extent_ == (2.0, 3.0)
    assert space.boundary == 'reflecting'
    assert np.array_equal(space.positions_, np.zeros((0, 2)))
