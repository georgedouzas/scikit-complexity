"""Test the `_regulation` module of `skcomplex.models.biology.physiology`."""

import math

import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.physiology import (
    coagulation_positive_feedback,
    glucose_dual_hormone_control,
)


def test_glucose_dual_hormone_control_attenuates_the_drop():
    """Test that the glucagon loop limits the drop of glucose during exercise, which insulin alone cannot."""
    dual = (
        Simulation(glucose_dual_hormone_control(), dt=0.05).check().prepare().run(24).trajectory_.values('organism.G')
    )
    single = (
        Simulation(glucose_dual_hormone_control(glucagon=False), dt=0.05)
        .check()
        .prepare()
        .run(24)
        .trajectory_.values('organism.G')
    )
    assert 1 - dual.min() < 0.8 * (1 - single.min())
    assert dual[-1] == pytest.approx(1.0, abs=1e-3)
    assert single[-1] == pytest.approx(1.0, abs=1e-3)


def test_coagulation_positive_feedback_is_all_or_nothing():
    """Test that a weak injury fades, a stronger one bursts to a similar peak, and fibrinolysis dissolves the clot."""
    peaks, clots = {}, {}
    for injury in (0.3, 0.7, 1.0):
        trajectory = (
            Simulation(coagulation_positive_feedback(injury=injury), dt=0.1).check().prepare().run(120).trajectory_
        )
        peaks[injury], clots[injury] = trajectory.values('blood.T').max(), trajectory.values('blood.F')
    threshold = (3 - math.sqrt(5)) / 2
    assert peaks[0.3] < threshold < peaks[0.7]
    assert peaks[1.0] < 1.2 * peaks[0.7]
    assert clots[1.0][-1] < 0.01 * clots[1.0].max()
