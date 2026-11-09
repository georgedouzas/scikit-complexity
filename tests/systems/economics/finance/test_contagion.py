"""Test the `_contagion` module of `skcomplex.systems.economics.finance`."""

import numpy as np
import pytest

from skcomplex.systems.economics.finance import (
    DebtrankSystem,
)

from ..conftest import _agents, _run


def test_debtrank_spreads_distress_once_through_the_loans():
    """Test DebtRank on two banks by hand, and that every bank ends inactive or undistressed in a random network."""
    pair = _run(
        DebtrankSystem([[0.0, 5.0], [0.0, 0.0]], [10.0, 10.0]).prepare().system_,
        5,
        observables=DebtrankSystem().get_run_settings()['observables'],
    )
    assert np.allclose(_agents(pair, 'banks', 5)['h'], [1.0, 0.5])
    assert pair.get_values('value_at_risk')[-1] - pair.get_values('value_at_risk')[0] == pytest.approx(0.5)
    network = _run(
        DebtrankSystem().prepare().system_,
        10,
        observables=DebtrankSystem().get_run_settings()['observables'],
    )
    assert np.all(np.diff(network.get_values('value_at_risk')) >= 0)
    assert network.get_values('value_at_risk')[-1] <= 1.0
    assert set(_agents(network, 'banks', 10)['status'].tolist()) <= {0.0, 2.0}
