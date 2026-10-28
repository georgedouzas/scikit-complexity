"""Test the `_traps` module of `skcomplex.models.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.statistical_mechanics import trap_model


def test_trap_model_escapes_grow_as_t_to_the_temperature():
    """Test that the escapes per walker below the glass temperature grow as t^T."""
    trajectory = Simulation(trap_model(5000, T=0.5), dt=100, random_state=0).check().run(100000).trajectory_
    late = trajectory.times > 1000
    escapes = trajectory.values('escapes.renewals')[late] / 5000
    slope = np.polyfit(np.log(trajectory.times[late]), np.log(escapes), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.05)
