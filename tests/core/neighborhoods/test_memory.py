"""Test the `_memory` module of `skcomplex.core.neighborhoods`."""

from skcomplex.core.neighborhoods import MemoryNeighborhood


def test_memory_neighbourhood_keeps_its_words_in_order():
    """Test that a memory set prepares its words as strings in the given order."""
    assert MemoryNeighborhood(['a*b', '1', 'a^-1']).check().prepare().elements_ == ('a*b', '1', 'a^-1')
