"""Test the `_generator_neighborhood` module of `skcomplex.systems.computer_science.automata`."""

from skcomplex.systems.computer_science.automata import GeneratorNeighborhood


def test_generator_neighbourhood_lists_the_identity_and_inverses():
    """Test that the memory set of the generators holds the identity first and every generator with its inverse."""
    assert GeneratorNeighborhood(('a', 'b')).prepare().elements_ == ('1', 'a', 'a^-1', 'b', 'b^-1')
    assert GeneratorNeighborhood(('a', 'b'), center=False, inverses=False).prepare().elements_ == ('a', 'b')
