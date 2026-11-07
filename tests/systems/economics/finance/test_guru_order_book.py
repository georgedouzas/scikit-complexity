"""Test the `_guru_order_book` module of `skcomplex.systems.economics.finance`."""

import numpy as np
from scipy.stats import kurtosis

from skcomplex.systems import describe
from skcomplex.systems.economics.finance import GuruOrderBookSystem

from ..conftest import _run_observed


def test_guru_order_book_coordinates_on_gurus_with_fat_tailed_returns():
    """Test that holdings are conserved, that a guru gathers most followers, and that returns have fat tails."""
    system = GuruOrderBookSystem().prepare().system_
    trajectory = _run_observed(
        system,
        1000,
        observables=describe('guru_order_book').observables_of(system),
        invariants={'total_cash': 1e-6, 'total_stocks': 1e-6},
    )
    returns = trajectory.values('market.log_return')[1:]
    assert trajectory.values('market.guru_followers')[1:].max() > 40
    assert kurtosis(returns) > 3
    beta = trajectory.values('market.beta')[1:]
    assert beta.max() > 2
    assert np.any(np.isclose(beta[1:], 1.0) & (beta[:-1] > 1.0))
