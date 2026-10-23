"""Test the `_contagion` module of `skcomplex.models.economics.finance`."""

import numpy as np
import pytest

from skcomplex.models import describe
from skcomplex.models.economics.finance import (
    debtrank,
)

from ..conftest import _agents, _run


def test_debtrank_spreads_distress_once_through_the_loans():
    """Test DebtRank on two banks by hand, and that every bank ends inactive or undistressed in a random network."""
    pair = _run(
        built := debtrank([[0.0, 5.0], [0.0, 0.0]], [10.0, 10.0]),
        5,
        observables=describe('debtrank').observables_of(built),
    )
    assert np.allclose(_agents(pair, 'banks', 5)['h'], [1.0, 0.5])
    assert pair['value_at_risk'][-1] - pair['value_at_risk'][0] == pytest.approx(0.5)
    network = _run(built := debtrank(), 10, observables=describe('debtrank').observables_of(built))
    assert np.all(np.diff(network['value_at_risk']) >= 0)
    assert network['value_at_risk'][-1] <= 1.0
    assert set(_agents(network, 'banks', 10)['status'].tolist()) <= {0.0, 2.0}
