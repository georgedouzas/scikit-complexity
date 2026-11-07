"""Test the `_regulatory_fba` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.systems_biology import RegulatoryFBASystem

THRESHOLD = 0.1


def test_glucose_represses_acetate_uptake():
    """Test that acetate is untouched while glucose lasts, and that biomass grows in two phases with a lag."""
    trajectory = simulate(RegulatoryFBASystem())
    times = trajectory.times
    glucose = trajectory.values('culture/glucose.c')
    biomass = trajectory.values('culture/biomass.x')
    uptake = -trajectory.values('control/cell.v_EX_ac_e')
    assert np.all(uptake[glucose > THRESHOLD] == pytest.approx(0.0, abs=1e-9))
    assert np.max(uptake[glucose <= THRESHOLD]) > 1
    growth = np.gradient(np.log(biomass), times)
    depleted = times[np.argmax(glucose <= 1e-3)]
    started = times[np.argmax(uptake > 1e-6)]
    lag = (times > depleted) & (times < started)
    assert started > depleted
    assert np.all(np.abs(growth[lag][1:-1]) < 0.01)
    assert biomass[times > started][-1] > biomass[lag][-1] * 1.05
