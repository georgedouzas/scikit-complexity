"""Test the `_coupled_colpitts` module of `skcomplex.systems.engineering.circuits`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.circuits import CoupledColpittsSystem


def _phase(y, z):
    """Return the unwrapped angle of an orbit in the plane of two variables, around their means."""
    return np.unwrap(np.arctan2(z - z.mean(), y - y.mean()))


@pytest.mark.parametrize(('k', 'locked'), [(0.0, False), (0.4, True)])
def test_coupled_colpitts_lock_their_phases_for_moderate_coupling(k, locked):
    """Test that the phases of the detuned oscillators drift apart without coupling and stay together for `k = 0.4`."""
    trajectory = simulate(CoupledColpittsSystem(k=k), until=400)
    settled = trajectory.times > 100
    phases = [
        _phase(
            trajectory.get_values(f'oscillator_{index}.y')[settled],
            trajectory.get_values(f'oscillator_{index}.z')[settled],
        )
        for index in range(2)
    ]
    assert (np.ptp(phases[0] - phases[1]) < 2 * math.pi) == locked
