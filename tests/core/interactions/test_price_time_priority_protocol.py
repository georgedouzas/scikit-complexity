"""Test the `_price_time_priority_protocol` module of `skcomplex.core.interactions`."""

from skcomplex.core.interactions import PriceTimePriorityProtocol


def test_match_gives_priority_to_the_earlier_offer(clear):
    """Test that of two providers at the same limit, the earlier one matches first."""
    protocol = PriceTimePriorityProtocol(arrival='sequential')
    matches = clear(protocol, {'limit': [10.0]}, {'limit': [10.0, 10.0], 'value': [0.0, 1.0]})
    assert matches == [(10.0, 10.0, 0.0, 1.0, 10.0)]


def test_match_trades_at_the_resting_limits(clear):
    """Test that an arriving request matches the lowest resting offers first, at their limits."""
    protocol = PriceTimePriorityProtocol(arrival='sequential')
    matches = clear(protocol, {'limit': [11.5], 'quantity': 3.0}, {'limit': [12.0, 9.0, 11.0]})
    assert [price for _, _, _, _, price in matches] == [9.0, 11.0]
