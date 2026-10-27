"""Test the times of a system: their checks, their prepared kind and schedule, and their frames."""

import pytest

from skcomplex.core.base import TimeKind, ValidationError
from skcomplex.core.spaces import ContinuousTime, DiscreteTime, StaticTime
from skcomplex.core.specs import ScheduleSpec


def test_times_fit_their_kind():
    """Test that every time prepares its kind, and only a discrete time has a schedule."""
    assert ContinuousTime().prepare().time_kind_ == TimeKind.CONTINUOUS
    assert StaticTime().prepare().schedule_ is None
    assert DiscreteTime().prepare().schedule_ == 'synchronous'


def test_discrete_time_checks_its_schedule():
    """Test that an unknown schedule is stored as given and rejected by the check."""
    time = DiscreteTime(schedule='parallel')
    assert time.schedule == 'parallel'
    with pytest.raises(ValidationError, match='schedule'):
        time.check()
    schedule = ScheduleSpec()
    assert DiscreteTime(schedule=schedule).prepare().schedule_ is schedule


def test_times_run_their_frames():
    """Test that discrete time steps by one, continuous time spans the run, and static time solves once."""
    assert DiscreteTime().prepare().run(2.5).tolist() == [0.0, 1.0, 2.0]
    assert ContinuousTime().prepare().run(3.0).tolist() == [0.0, 3.0]
    assert StaticTime().prepare().run(3.0).tolist() == [0.0]
