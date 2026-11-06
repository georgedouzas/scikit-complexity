"""Test the `_nearest` module of `skcomplex.core.components.spaces`."""

import pytest

from skcomplex.core.components.spaces import NearestNeighborhood
from skcomplex.core.errors import ValidationError


@pytest.mark.parametrize('k', [0, 2.5])
def test_nearest_neighbourhood_needs_a_whole_number_of_neighbours(k):
    """Test that a nearest neighbourhood rejects no neighbours and a fractional number of neighbours."""
    with pytest.raises(ValidationError, match='whole number'):
        NearestNeighborhood(k).prepare()
