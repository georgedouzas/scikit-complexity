"""Test the `_random_search_protocol` module of `skcomplex.core.interactions`."""

import pytest

from skcomplex.core.interactions import RandomSearchProtocol


def test_match_prices_at_the_midpoint_within_the_budget(clear):
    """Test that random search matches at the midpoint of the limits, as much as the budget affords."""
    matches = clear(RandomSearchProtocol(tries=2), {'limit': [10.0], 'budget': 3.0}, {'limit': [2.0]})
    assert matches[0][4] == 6.0
    assert matches[0][3] == pytest.approx(0.5)


def test_match_skips_limits_that_do_not_cross(clear):
    """Test that a requester whose limit is below the provider's limit does not match."""
    assert clear(RandomSearchProtocol(tries=3), {'limit': [1.0]}, {'limit': [2.0]}) == []
