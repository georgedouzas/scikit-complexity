"""Test the `_circulation` module of `skcomplex.systems.biology.physiology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.physiology import CardiovascularSystem


def test_cardiovascular_conserves_blood_and_pulses():
    """Test that the circulation conserves the volume of blood and gives an arterial pressure near 120/80 mmHg."""
    trajectory = simulate(CardiovascularSystem(), dt=0.02, until=6.4)
    total = (
        trajectory.get_values('ventricle.V') + trajectory.get_values('arteries.V') + trajectory.get_values('veins.V')
    )
    assert np.ptp(total) < 1e-6 * total[0]
    pressure = (trajectory.get_values('arteries.V')[trajectory.times > 4.8] - 250.0) * 1.3**-1
    assert 65 < pressure.min() < 95
    assert 105 < pressure.max() < 135
