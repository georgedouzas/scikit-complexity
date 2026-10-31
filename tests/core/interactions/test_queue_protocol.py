"""Test the `_queue_protocol` module of `skcomplex.core.interactions`."""

import pytest

from skcomplex.core.interactions import QueueProtocol
from skcomplex.errors import ValidationError


def test_match_serves_the_highest_limits_at_the_lowest_providers(clear):
    """Test that a queue by limit serves the highest requesters at the limits of the lowest providers."""
    matches = clear(QueueProtocol(order='limit'), {'limit': [10.0, 9.0, 1.0]}, {'limit': [4.0, 6.0]})
    assert [(requester, price) for requester, _, _, _, price in matches] == [(10.0, 4.0), (9.0, 6.0)]


def test_check_rejects_an_unknown_order():
    """Test that an unknown order of the queue raises an error."""
    with pytest.raises(ValidationError, match='order'):
        QueueProtocol(order='bid').check()
