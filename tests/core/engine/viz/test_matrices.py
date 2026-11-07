"""Test the `_matrices` module of `skcomplex.core.engine.viz`."""

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import RecorderObserver, Simulation
from skcomplex.core.engine.viz import MatrixVisual, RerunObserver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.computer_science.neural_networks import ElmanNetworkSystem


def _views(blueprint):
    """Return the views of a blueprint in layout order."""
    found, stack = [], [blueprint.root_container]
    while stack:
        part = stack.pop(0)
        if isinstance(part, rrb.View):
            found.append(part)
        else:
            stack = list(part.contents) + stack
    return found


def test_matrix_visual_logs_the_recurrent_weights(tmp_path):
    """Test that the recurrent weights are a 4 by 4 image at every frame whose values change while training."""
    recorder = RecorderObserver()
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    simulation = Simulation(
        ElmanNetworkSystem(n_bits=30).prepare().system_,
        observers=[observer, recorder],
        random_state=0,
    )
    trajectory = simulation.prepare().run(until=3).trajectory_
    visual = next(item for item in observer.visuals_ if isinstance(item, MatrixVisual) and item.variable == 'A')
    assert (visual.name_, visual.shape_) == ('net/recurrence.A', (4, 4))
    values = trajectory.values('net/recurrence.A')
    assert not np.array_equal(values[0], values[-1])
    assert observer.logged_frames_ == 4


def test_matrix_visual_rejects_a_variable_that_is_not_a_matrix():
    """Test that a variable without two dimensions raises an error."""
    problem = PassCompiler().prepare(ElmanNetworkSystem(n_bits=30).prepare().system_).run().problem_
    with pytest.raises(ValidationError, match='no matrix'):
        MatrixVisual('b', path='net/hidden', recording=rr.RecordingStream('test')).prepare(problem)


def test_first_view_of_a_system_that_trains():
    """Test that the first view of a system that trains is the time series of its loss and metrics over epochs."""
    observer = RerunObserver(blueprint='default').prepare(
        PassCompiler().prepare(ElmanNetworkSystem(n_bits=30).prepare().system_).run().problem_,
    )
    first = _views(observer.blueprint_)[0]
    assert isinstance(first, rrb.TimeSeriesView)
    assert str(first.origin) == '/reporters/training'


def test_default_blueprint_of_a_system_that_trains():
    """Test that the default blueprint holds the loss and one heat map per trained matrix, and no inner state."""
    observer = RerunObserver(blueprint='default').prepare(
        PassCompiler().prepare(ElmanNetworkSystem(n_bits=30).prepare().system_).run().problem_,
    )
    origins = [str(view.origin) for view in _views(observer.blueprint_)]
    assert origins == ['/reporters/training', 'net/input/D', 'net/recurrence/A', 'net/readout/W']
