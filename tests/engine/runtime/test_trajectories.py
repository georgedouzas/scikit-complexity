"""Test reading the recorded values of a trajectory by name."""

import pytest

from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.experiments import BaseExperiment
from skcomplex.models import create
from skcomplex.models.biology.ecology import lotka_volterra


def test_trajectory_reads_values_by_name():
    """Test that a trajectory returns the values of a state by name and is not indexed."""
    trajectory = Simulation(create('lotka_volterra'), dt=0.1).prepare().run(until=1.0).trajectory_
    values = trajectory.values('prey.n')
    assert values.shape[0] == trajectory.times.size
    with pytest.raises(TypeError):
        trajectory['prey.n']


def test_member_keeps_the_parameters_of_interactions():
    """Test that a member of a replicated system reads its own states and the shared parameters of interactions."""
    ensemble = BaseExperiment._replicate(lotka_volterra(), 2, {'prey.n': [20.0, 40.0]})
    trajectory = Simulation(ensemble, dt=0.1).prepare().run(until=1.0).trajectory_
    member = trajectory.member(1, compile_system(lotka_volterra()))
    assert float(member.values('prey.n')[0]) == 40.0
    assert float(member.values('growth.r')[0]) == 1.1
