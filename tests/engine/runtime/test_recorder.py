"""Test the `_recorder` module of `skcomplex.engine.runtime`."""

import pytest

from skcomplex.core.errors import SimulationError
from skcomplex.core.system import System
from skcomplex.engine.runtime import RecorderObserver, Simulation
from skcomplex.models.biology.ecology import PopulationElement


def test_recorder_builds_the_trajectory_of_its_frames():
    """Test that a recorder keeps every frame of a run and reads them as a trajectory."""
    recorder = RecorderObserver()
    Simulation(System(elements=[PopulationElement('a', n=2)]), dt=1, observers=[recorder]).run(until=2)
    assert len(recorder.frames_) == 3
    assert recorder.trajectory_.times.tolist() == [0.0, 1.0, 2.0]
    assert recorder.trajectory_.values('a.n').tolist() == [2.0, 2.0, 2.0]


def test_recorder_without_a_problem_raises():
    """Test that reading the trajectory of a recorder that never ran raises an error."""
    with pytest.raises(SimulationError, match='has not received a problem'):
        _ = RecorderObserver().trajectory_
