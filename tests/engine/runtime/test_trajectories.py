"""Test reading the recorded values of a trajectory by name."""

import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models import create


def test_trajectory_reads_values_by_name():
    """Test that a trajectory returns the values of a state by name and is not indexed."""
    trajectory = Simulation(create('lotka_volterra'), dt=0.1).check().run(until=1.0).trajectory_
    values = trajectory.values('prey.n')
    assert values.shape[0] == trajectory.times.size
    with pytest.raises(TypeError):
        trajectory['prey.n']
