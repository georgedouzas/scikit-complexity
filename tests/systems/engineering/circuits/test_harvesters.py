"""Test the `_harvesters` module of `skcomplex.systems.engineering.circuits`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.circuits import (
    VibrationEnergyHarvesterSystem,
)


def test_vibration_energy_harvester_steady_response():
    """Test that the sprung mass settles on the amplitude of the linear response, and the voltage follows the stroke."""
    trajectory = simulate(
        VibrationEnergyHarvesterSystem().prepare().system_,
        until=30,
        dt=0.01,
        observables=VibrationEnergyHarvesterSystem().get_run_settings()['observables'],
    )
    m1, m2, k1, k2, b, A, omega = 40.0, 400.0, 2.0e5, 2.0e4, 1500.0, 0.01, 9.42
    stiffness = np.array(
        [
            [k1 + k2 - m1 * omega**2 + 1j * omega * b, -k2 - 1j * omega * b],
            [-k2 - 1j * omega * b, k2 - m2 * omega**2 + 1j * omega * b],
        ],
    )
    amplitude = abs(np.linalg.solve(stiffness, [k1 * A, 0.0])[1])
    sprung = trajectory.values('harvester.y2')[-1000:]
    assert np.ptp(sprung) / 2 == pytest.approx(amplitude, rel=0.01)
    stroke = trajectory.values('harvester.y2') - trajectory.values('harvester.y1')
    assert trajectory.values('voltage') == pytest.approx(10.0 * 4.0e-10 * 2.0e4 * stroke / 1.0e-8)
