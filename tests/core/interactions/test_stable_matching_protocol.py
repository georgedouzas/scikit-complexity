"""Test the `_stable_matching_protocol` module of `skcomplex.core.interactions`."""

import numpy as np

from skcomplex.core.interactions import StableMatchingProtocol


def _score(first, second):
    """Return the preference of a value for another, higher for a nearer and then a lower value."""
    return -abs(first - second) - 0.01 * second


def _closeness(first, second):
    """Return the preference of a participant for another by their values."""
    return _score(first.value, second.value)


def test_match_gives_the_preferred_provider_to_its_preferred_requester(clear):
    """Test that the lowest provider goes to the highest requester, which it prefers by default."""
    matches = clear(StableMatchingProtocol(), {'limit': [10.0, 12.0]}, {'limit': [5.0, 7.0]})
    assert {(requester, provider) for requester, provider, _, _, _ in matches} == {(12.0, 5.0), (10.0, 7.0)}


def test_match_leaves_no_blocking_pair(clear):
    """Test that no unmatched requester and provider both prefer each other to their partners."""
    rng = np.random.default_rng(3)
    requesters, providers = rng.uniform(0, 10, 6), rng.uniform(0, 10, 6)
    protocol = StableMatchingProtocol(requester_preference=_closeness, provider_preference=_closeness)
    limits = 1.0 + np.arange(6)
    matches = clear(protocol, {'limit': limits, 'value': requesters}, {'limit': np.zeros(6), 'value': providers})
    by_limit = dict(zip(limits, requesters, strict=True))
    partners = {by_limit[requester]: value for requester, _, value, _, _ in matches}
    reverse = {value: requester for requester, value in partners.items()}
    assert len(partners) == 6

    for requester in requesters:
        for provider in providers:
            if partners[requester] == provider:
                continue
            requester_prefers = _score(requester, provider) > _score(requester, partners[requester])
            provider_prefers = _score(provider, requester) > _score(provider, reverse[provider])
            assert not (requester_prefers and provider_prefers)
