"""Test the `_gauge_arbitrage` module of `skcomplex.models.economics.finance`."""

import numpy as np

from skcomplex.models import run


def test_market_conserves_the_good_and_the_gauss_law():
    """Test that the total good stays within 1e-10 relative error, and the Gauss residual below 1e-8."""
    trajectory = run('gauge_arbitrage_market').trajectory_
    total = trajectory.values('total_good')
    assert np.abs(total - total[0]).max() / total[0] < 1e-10
    assert trajectory.values('gauss_residual').max() < 1e-8


def test_market_moves_the_good_and_the_exchange_rates():
    """Test that the good spreads out and the potentials leave their initial value."""
    trajectory = run('gauge_arbitrage_market').trajectory_
    density = trajectory.values('sites.rho')
    assert density[-1].std() < density[0].std()
    assert np.abs(trajectory.values('links.A')[-1]).max() > 0.01
