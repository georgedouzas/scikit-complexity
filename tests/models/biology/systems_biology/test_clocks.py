"""Test the `_clocks` module of `skcomplex.models.biology.systems_biology`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.systems_biology import (
    circadian_clock,
)


def _crossing_periods(times, series, level):
    """Return the intervals between upward crossings of a level."""
    above = np.asarray(series) >= level
    crossings = np.flatnonzero(~above[:-1] & above[1:])
    return np.diff(np.asarray(times)[crossings])


def test_circadian_clock_period_is_about_a_day():
    """Test that Goldbeter's PER model oscillates with a period of about 23.7 hours."""
    trajectory = Simulation(circadian_clock(), dt=0.5).check().prepare().run(until=240).trajectory_
    keep = trajectory.times > 48
    mrna = trajectory.values('clock.M')[keep]
    periods = _crossing_periods(trajectory.times[keep], mrna, mrna.mean())
    assert abs(periods.mean() - 23.7) < 1.0


def test_circadian_clock_period_lengthens_with_degradation():
    """Test that a larger maximal degradation rate of PER lengthens the period of Goldbeter's clock."""
    periods = []
    for v_d in (0.8, 1.6):
        trajectory = Simulation(circadian_clock(v_d=v_d), dt=0.1).check().prepare().run(until=480).trajectory_
        keep = trajectory.times > 240
        mrna = trajectory.values('clock.M')[keep]
        periods.append(_crossing_periods(trajectory.times[keep], mrna, mrna.mean()).mean())
    assert periods[0] == pytest.approx(22.6, abs=0.5)
    assert periods[1] == pytest.approx(29.3, abs=0.5)
