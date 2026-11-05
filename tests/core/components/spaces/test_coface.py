"""Test the `_coface` module of `skcomplex.core.components.spaces`."""

from skcomplex.core.components.spaces import CofaceNeighborhood


def test_coface_neighborhood_prepares_with_its_label():
    """Test that a coface neighbourhood keeps the label it is given."""
    assert CofaceNeighborhood(label='faces').prepare().label == 'faces'
