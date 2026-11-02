"""Test the `_arbitrage` module of `skcomplex.models.economics.finance`."""

import numpy as np
import pytest

from skcomplex.models.economics.finance import (
    arbitrage_traders,
)

from ..conftest import _run


@pytest.mark.parametrize(('E', 'shift'), [(0.0, 0.0), (0.5, 1.0)])
def test_arbitrage_traders_drift_only_with_an_arbitrage_opportunity(E, shift):
    """Test that the good diffuses in place without arbitrage and drifts by `D E t^2` with it."""
    density = _run(arbitrage_traders(E=E), 2.0, dt=0.1).values('market.rho')[-1]
    positions = (np.arange(200) + 0.5) * 0.1 - 10
    assert (density * positions).sum() / density.sum() == pytest.approx(shift, abs=1e-3)
