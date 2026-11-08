"""Test the `_traps` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import TrapSystem


def test_trap_system_escapes_grow_as_t_to_the_temperature():
    """Test that the escapes per walker below the glass temperature grow as t^T."""
    trajectory = simulate(TrapSystem(5000, T=0.5), until=100000, dt=100, random_state=0)
    late = trajectory.times > 1000
    escapes = trajectory.values('escapes.renewals')[late] / 5000
    slope = np.polyfit(np.log(trajectory.times[late]), np.log(escapes), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.05)
