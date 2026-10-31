"""Test the `_uniform_price_protocol` module of `skcomplex.core.interactions`."""

from skcomplex.core.interactions import UniformPriceProtocol


def test_match_pairs_the_crossing_offers_at_one_price(clear):
    """Test that the highest requesters match the lowest providers at the midpoint of the last crossing pair."""
    matches = clear(UniformPriceProtocol(), {'limit': [10.0, 8.0, 3.0]}, {'limit': [2.0, 5.0, 9.0]})
    assert len(matches) == 2
    assert {price for _, _, _, _, price in matches} == {6.5}


def test_match_matches_nothing_without_crossing_limits(clear):
    """Test that limits that never cross give no matches."""
    assert clear(UniformPriceProtocol(), {'limit': [1.0]}, {'limit': [2.0]}) == []
