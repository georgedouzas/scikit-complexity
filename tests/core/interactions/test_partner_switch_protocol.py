"""Test the `_partner_switch_protocol` module of `skcomplex.core.interactions`."""

from skcomplex.core.interactions import PartnerSwitchProtocol


def test_match_settles_on_the_lowest_partner(clear):
    """Test that a requester switches to the lower provider and then keeps it as its partner."""
    protocol = PartnerSwitchProtocol(sample=1, intensity=100.0)
    requesters = {'limit': [10.0], 'quantity': 100.0, 'budget': 4.0}
    matches = clear(protocol, requesters, {'limit': [5.0, 4.0], 'quantity': 100.0}, steps=20)
    assert [provider for _, provider, _, _, _ in matches[-5:]] == [4.0] * 5


def test_match_prices_at_the_partner_limit(clear):
    """Test that a requester matches at the limit of its partner."""
    matches = clear(PartnerSwitchProtocol(sample=2), {'limit': [10.0]}, {'limit': [5.0, 4.0]})
    assert matches == [(10.0, 4.0, 0.0, 1.0, 4.0)]
