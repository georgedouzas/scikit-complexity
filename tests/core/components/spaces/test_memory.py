"""Test the `_memory` module of `skcomplex.core.components.spaces`."""

from skcomplex.core.components.spaces import MemoryNeighborhood


def test_memory_neighbourhood_keeps_its_words_in_order():
    """Test that a memory set prepares its words as strings in the given order."""
    assert MemoryNeighborhood(['a*b', '1', 'a^-1']).prepare().elements_ == ('a*b', '1', 'a^-1')
