"""Test the `_recorder` module of `skcomplex.engine.runtime`."""

import numpy as np
import pytest

from skcomplex.core.system import System
from skcomplex.engine.runtime import RecorderObserver, Simulation
from skcomplex.errors import SimulationError
from skcomplex.models import create
from skcomplex.models.biology.common.populations import PopulationElement


def test_recorder_builds_the_trajectory_of_its_frames():
    """Test that a recorder keeps every frame of a run and reads them as a trajectory."""
    recorder = RecorderObserver()
    Simulation(System(elements=[PopulationElement('a', n=2)]), dt=1, observers=[recorder]).check().prepare().run(
        until=2,
    )
    assert len(recorder.frames_) == 3
    assert recorder.trajectory_.times.tolist() == [0.0, 1.0, 2.0]
    assert recorder.trajectory_.values('a.n').tolist() == [2.0, 2.0, 2.0]


def test_recorder_keeps_one_row_of_parameters_that_stay_the_same():
    """Test that frames share unchanged parameters, so large constant parameters take no memory per frame."""
    recorder = RecorderObserver()
    Simulation(create('sir'), dt=1, observers=[recorder]).check().prepare().run(until=2)
    parameters = recorder.trajectory_.parameters
    assert parameters.shape[0] == 3
    assert parameters.shape[1] > 0
    assert np.shares_memory(parameters[0], parameters[2])


def test_recorder_without_a_problem_raises():
    """Test that reading the trajectory of a recorder that never ran raises an error."""
    with pytest.raises(SimulationError, match='has not received a problem'):
        _ = RecorderObserver().trajectory_
