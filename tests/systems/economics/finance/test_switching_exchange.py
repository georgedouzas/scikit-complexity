"""Test the `_switching_exchange` module of `skcomplex.systems.economics.finance`."""

import numpy as np
import pytest

from skcomplex.systems.economics.finance import KirmanExchangeRateSystem

from ..conftest import _run


def test_kirman_exchange_rate_follows_eq_1_37_with_fundamentalists_only():
    """Test the eigenvalue `(1 - v) / (1 + r)` of a market of fundamentalists without shocks or switching."""
    trajectory = _run(KirmanExchangeRateSystem(k=30, sigma=0.0, epsilon=0.0), 5)
    assert trajectory.get_values('market.x')[1:] == pytest.approx(0.1 * (0.5 / 1.01) ** np.arange(1, 6))


def test_kirman_exchange_rate_is_bimodal_and_more_volatile_with_chartists():
    """Test that the fundamentalists are mostly near the extremes and that chartists make the rate volatile."""
    system = KirmanExchangeRateSystem().prepare().system_
    trajectory = _run(system, 40000, observables=KirmanExchangeRateSystem().get_run_settings()['observables'])
    share, deviation = trajectory.get_values('market.fundamentalists'), trajectory.get_values('market.x')
    assert np.mean((share < 0.2) | (share > 0.8)) > 0.5
    changes = np.abs(np.diff(deviation))
    assert changes[share[1:] < 0.2].mean() > 2 * changes[share[1:] > 0.8].mean()
