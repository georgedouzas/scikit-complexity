"""Test the `_circulation` module of `skcomplex.models.biology.physiology`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.physiology import (
    cardiovascular,
)


def test_cardiovascular_conserves_blood_and_pulses():
    """Test that the circulation conserves the volume of blood and gives an arterial pressure near 120/80 mmHg."""
    trajectory = Simulation(cardiovascular(), dt=0.02).run(until=6.4).trajectory_
    total = trajectory['ventricle.V'] + trajectory['arteries.V'] + trajectory['veins.V']
    assert np.ptp(total) < 1e-6 * total[0]
    pressure = (trajectory['arteries.V'][trajectory.times > 4.8] - 250.0) * 1.3**-1
    assert 65 < pressure.min() < 95
    assert 105 < pressure.max() < 135
