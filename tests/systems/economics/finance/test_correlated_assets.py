"""Test the `_correlated_assets` module of `skcomplex.systems.economics.finance`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.systems.economics.finance import (
    CorrelatedGeometricBrownianMotionSystem,
)


def test_correlated_geometric_brownian_motion_correlates_returns():
    """Test that the log returns per frame of the two assets have the sample correlation 0.7 within 0.05."""
    system = CorrelatedGeometricBrownianMotionSystem(rho=0.7).prepare().system_
    trajectory = simulate(system, until=100, solver=EulerMaruyamaSolver(dt=1e-3), dt=0.01, random_state=0)
    first, second = (np.diff(np.log(trajectory.values(f'{label}.price'))) for label in ('first', 'second'))
    assert first.size == 10_000
    assert np.corrcoef(first, second)[0, 1] == pytest.approx(0.7, abs=0.05)
