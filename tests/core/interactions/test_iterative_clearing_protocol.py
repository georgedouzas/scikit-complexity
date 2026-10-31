"""Test the `_iterative_clearing_protocol` module of `skcomplex.core.interactions`."""

from skcomplex.core.interactions import IterativeClearingProtocol


def test_match_clears_at_a_price_between_the_limits(clear):
    """Test that the tâtonnement matches every crossing pair at one price between the marginal limits."""
    matches = clear(IterativeClearingProtocol(), {'limit': [10.0, 9.0, 1.0]}, {'limit': [4.0, 6.0]})
    assert len(matches) == 2
    assert len({price for _, _, _, _, price in matches}) == 1
    assert all(6.0 <= price <= 9.0 for _, _, _, _, price in matches)


def test_match_matches_nothing_without_positive_limits(clear):
    """Test that offers without a positive limit do not match."""
    assert clear(IterativeClearingProtocol(), {'limit': [0.0]}, {'limit': [0.0]}) == []
