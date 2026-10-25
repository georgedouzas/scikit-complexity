"""Test the `_grids` module of `skcomplex.models.engineering.power_systems`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.engineering.power_systems import (
    power_grid,
)


@pytest.mark.parametrize(('changes', 'synchronous'), [({}, True), ({'added_line': (0, 2)}, False)])
def test_power_grid_braess_paradox(changes, synchronous):
    """Test that the ring grid synchronises, and loses synchrony when a line is added."""
    trajectory = Simulation(power_grid(**changes), dt=1.0).run(until=100).trajectory_
    frequencies = np.array([trajectory.values(f'node_{index}.phi_d1')[-10:] for index in range(8)])
    assert (np.abs(frequencies).max() < 0.1) == synchronous
