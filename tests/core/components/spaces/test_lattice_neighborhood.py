"""Test the `_lattice_neighborhood` module of `skcomplex.core.components.spaces`."""

import pytest

from skcomplex.core.components.spaces import LatticeNeighborhood
from skcomplex.core.errors import ValidationError


def test_lattice_neighbourhood_keeps_the_order_of_its_offsets():
    """Test that a lattice neighbourhood prepares its offsets as tuples of integers in the given order."""
    assert LatticeNeighborhood([(1, 0), [0, -1]]).prepare().offsets_ == ((1, 0), (0, -1))


def test_lattice_neighbourhood_rejects_offsets_of_different_dimensions():
    """Test that offsets of different numbers of dimensions are rejected by the check."""
    with pytest.raises(ValidationError, match='same number of dimensions'):
        LatticeNeighborhood([(1, 0), (1,)]).prepare()
