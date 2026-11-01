"""Test the `_double_auction` module of `skcomplex.models.economics.markets`."""

import pytest

from skcomplex.models.economics.markets import double_auction

from ..conftest import _run_observed


@pytest.mark.parametrize('protocol', ['order_book', 'call_auction'])
def test_double_auction_budget_constraint_is_efficient(protocol):
    """Zero-intelligence traders with budget constraints realise over 90% of the surplus, without them much less."""
    constrained = _run_observed(double_auction(30, 30, protocol=protocol), 10).values('ledger.efficiency')[1:]
    unconstrained = _run_observed(double_auction(30, 30, protocol=protocol, constrained=False), 10).values(
        'ledger.efficiency',
    )[1:]
    assert constrained.mean() > 0.9
    assert unconstrained.mean() < constrained.mean() - 0.1


def test_double_auction_rejects_unknown_protocol():
    """An unknown protocol raises an error."""
    with pytest.raises(ValueError, match='protocol'):
        double_auction(protocol='barter')
