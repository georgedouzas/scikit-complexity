"""Test the `_double_auction` module of `skcomplex.systems.economics.markets`."""

import numpy as np
import pytest

from skcomplex.systems.economics.markets import DoubleAuctionSystem, maximum_surplus

from ..conftest import _run_observed


@pytest.mark.parametrize('protocol', ['order_book', 'call_auction'])
def test_double_auction_budget_constraint_is_efficient(protocol):
    """Zero-intelligence traders with budget constraints realise over 90% of the surplus, without them much less."""
    constrained = _run_observed(DoubleAuctionSystem(30, 30, protocol=protocol).prepare().system_, 10).values(
        'ledger.efficiency',
    )[1:]
    unconstrained = _run_observed(
        DoubleAuctionSystem(30, 30, protocol=protocol, constrained=False).prepare().system_,
        10,
    ).values('ledger.efficiency')[1:]
    assert constrained.mean() > 0.9
    assert unconstrained.mean() < constrained.mean() - 0.1


def test_double_auction_rejects_unknown_protocol():
    """An unknown protocol raises an error."""
    with pytest.raises(ValueError, match='protocol'):
        DoubleAuctionSystem(protocol='barter').prepare()


def test_maximum_surplus():
    """The maximum surplus pairs the highest values with the lowest costs."""
    assert maximum_surplus(np.array([10.0, 6.0, 2.0]), np.array([1.0, 4.0, 8.0])) == 11.0
