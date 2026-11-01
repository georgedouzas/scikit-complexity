"""Test the `_lattice_neighborhood` module of `skcomplex.core.spaces` and the lattice neighbourhoods built on it."""

import pytest

from skcomplex.core.spaces import (
    BoxNeighborhood,
    LatticeNeighborhood,
    MooreNeighborhood,
    TriangularNeighborhood,
    VonNeumannNeighborhood,
)
from skcomplex.errors import ValidationError


def test_lattice_neighbourhood_keeps_the_order_of_its_offsets():
    """Test that a lattice neighbourhood prepares its offsets as tuples of integers in the given order."""
    assert LatticeNeighborhood([(1, 0), [0, -1]]).check().prepare().offsets_ == ((1, 0), (0, -1))


def test_lattice_neighbourhood_rejects_offsets_of_different_dimensions():
    """Test that offsets of different numbers of dimensions are rejected by the check."""
    with pytest.raises(ValidationError, match='same number of dimensions'):
        LatticeNeighborhood([(1, 0), (1,)]).check()


@pytest.mark.parametrize(
    ('neighborhood', 'size'),
    [
        (MooreNeighborhood(), 8),
        (MooreNeighborhood(center=True), 9),
        (VonNeumannNeighborhood(), 4),
        (VonNeumannNeighborhood(radius=2, center=True), 13),
        (BoxNeighborhood(2), 5),
        (TriangularNeighborhood(center=True), 7),
    ],
)
def test_lattice_neighbourhoods_list_their_offsets(neighborhood, size):
    """Test that every lattice neighbourhood prepares its number of offsets, the site itself first when included."""
    offsets = neighborhood.check().prepare().offsets_
    assert len(offsets) == size
    if getattr(neighborhood, 'center', False):
        assert not any(offsets[0])
