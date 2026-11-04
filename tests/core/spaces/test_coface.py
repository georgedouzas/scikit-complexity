"""Test the `_coface` module of `skcomplex.core.spaces`."""

from skcomplex.core.spaces import CofaceNeighborhood


def test_coface_neighborhood_prepares_with_its_label():
    """Test that a coface neighbourhood keeps the label it is given."""
    assert CofaceNeighborhood(label='faces').check().prepare().label == 'faces'
