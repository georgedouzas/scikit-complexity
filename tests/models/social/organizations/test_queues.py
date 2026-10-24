"""Test the `_queues` module of `skcomplex.models.social.organizations`."""

import numpy as np
import pytest

from skcomplex.models.social.organizations import (
    priority_queues,
)

from ..conftest import _run


def _waits(trajectory, frames):
    """Return the waiting times of the tasks executed in a run of the priority queue."""
    waits = []
    for frame in range(1, frames + 1):
        values = trajectory.agents('tasks', frame).values
        waits += values['wait'][values['age'] == 0].tolist()
    return np.array(waits)


def test_priority_queue_waits():
    """Test that random execution gives a mean wait of two for two tasks, and greedy execution gives long waits."""
    random = _waits(_run(priority_queues(greed=0.0, random_state=0), 2000), 2000)
    assert random.mean() == pytest.approx(2.0, rel=0.1)
    greedy = _waits(_run(priority_queues(random_state=0), 2000), 2000)
    assert np.median(greedy) == 1
    assert greedy.max() > 100


def test_priority_queue_reports_every_wait():
    """Test that the execution reports the wait of every executed task as a series of the trajectory."""
    trajectory = _run(priority_queues(random_state=0), 500)
    np.testing.assert_array_equal(trajectory['execution.wait'][1:], _waits(trajectory, 500))
