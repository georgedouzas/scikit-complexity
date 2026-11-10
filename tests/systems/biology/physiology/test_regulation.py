"""Test the `_regulation` module of `skcomplex.systems.biology.physiology`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.physiology import (
    CoagulationPositiveFeedbackSystem,
    GlucoseDualHormoneControlSystem,
    SteerableFeedbackLoopSystem,
)


def test_glucose_dual_hormone_control_attenuates_the_drop():
    """Test that the glucagon loop limits the drop of glucose during exercise, which insulin alone cannot."""
    dual = simulate(GlucoseDualHormoneControlSystem(), dt=0.05, until=24).get_values('organism.G')
    single = simulate(GlucoseDualHormoneControlSystem(glucagon=False), dt=0.05, until=24).get_values('organism.G')
    assert 1 - dual.min() < 0.8 * (1 - single.min())
    assert dual[-1] == pytest.approx(1.0, abs=1e-3)
    assert single[-1] == pytest.approx(1.0, abs=1e-3)


def test_coagulation_positive_feedback_is_all_or_nothing():
    """Test that a weak injury fades, a stronger one bursts to a similar peak, and fibrinolysis dissolves the clot."""
    peaks, clots = {}, {}
    for injury in (0.3, 0.7, 1.0):
        trajectory = simulate(CoagulationPositiveFeedbackSystem(injury=injury), dt=0.1, until=120)
        peaks[injury], clots[injury] = trajectory.get_values('blood.T').max(), trajectory.get_values('blood.F')
    threshold = (3 - math.sqrt(5)) / 2
    assert peaks[0.3] < threshold < peaks[0.7]
    assert peaks[1.0] < 1.2 * peaks[0.7]
    assert clots[1.0][-1] < 0.01 * clots[1.0].max()


def test_steering_signal_shifts_the_set_point_and_releases_it():
    """Test that the loop holds `x = 1`, climbs to the steered set point while the signal acts, and returns after."""
    trajectory = simulate(SteerableFeedbackLoopSystem(), dt=0.5, until=60)
    product, times = trajectory.get_values('loop.x'), trajectory.times
    assert product[(times > 15) & (times < 20)] == pytest.approx(1.0, abs=1e-3)
    assert product[(times > 35) & (times < 40)] == pytest.approx(1.2911656, abs=1e-3)
    assert product[times > 55] == pytest.approx(1.0, abs=1e-3)


def test_a_long_transmission_delay_makes_the_loop_oscillate():
    """Test that a transmission delay long against the response of the loop turns regulation into oscillations."""
    trajectory = simulate(SteerableFeedbackLoopSystem(tau=3.0, steering=0.0).prepare().system_, until=60, dt=0.5)
    late = trajectory.get_values('loop.x')[60:]
    assert np.ptp(late) > 0.8
