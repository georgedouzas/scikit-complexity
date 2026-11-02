"""Test the `_bass` module of `skcomplex.models.economics.innovation`."""

import numpy as np
import pytest

from skcomplex.models.economics.innovation import bass_diffusion

from ...conftest import _run_monitored


def test_bass_diffusion_matches_closed_form():
    """Test that the adopters follow the closed-form Bass curve and that the market is conserved."""
    p, q, market = 0.03, 0.38, 1000.0
    trajectory = _run_monitored(bass_diffusion(market, p=p, q=q), 0.25, 20)
    times = np.asarray(trajectory.times)
    decay = np.exp(-(p + q) * times)
    expected = market * (1 - decay) / (1 + q / p * decay)
    np.testing.assert_allclose(trajectory.values('market.A'), expected, atol=1e-2 * market)
    np.testing.assert_allclose(trajectory.values('market.A') + trajectory.values('market.P'), market)
    peak = times[np.argmax(trajectory.values('market.adoption'))]
    assert peak == pytest.approx(np.log(q / p) / (p + q), abs=0.25)
