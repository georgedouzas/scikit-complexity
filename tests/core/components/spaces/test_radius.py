"""Test the `_radius` module of `skcomplex.core.components.spaces`."""

import pytest

from skcomplex.core.components.spaces import RadiusNeighborhood
from skcomplex.core.errors import ValidationError


def test_radius_neighbourhood_needs_a_positive_radius():
    """Test that a radius neighbourhood takes a positive radius and rejects zero."""
    assert RadiusNeighborhood(1.5).radius == 1.5
    with pytest.raises(ValidationError, match='positive radius'):
        RadiusNeighborhood(0.0).prepare()
