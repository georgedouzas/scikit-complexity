"""Test the times of a system: their checks, their fitted kind and schedule, and their frames."""

import pytest

from skcomplex.core.base import TimeKind, ValidationError
from skcomplex.core.space import ContinuousTime, DiscreteTime, Schedule, StaticTime


def test_times_fit_their_kind():
    """Test that every time fits its kind, and only a discrete time has a schedule."""
    assert ContinuousTime().fit().time_kind_ == TimeKind.CONTINUOUS
    assert StaticTime().fit().schedule_ is None
    assert DiscreteTime().fit().schedule_ == 'synchronous'


def test_discrete_time_checks_its_schedule():
    """Test that an unknown schedule is stored as given and rejected by the check."""
    time = DiscreteTime(schedule='parallel')
    assert time.schedule == 'parallel'
    with pytest.raises(ValidationError, match='schedule'):
        time.check()
    schedule = Schedule()
    assert DiscreteTime(schedule=schedule).fit().schedule_ is schedule


def test_times_run_their_frames():
    """Test that discrete time steps by one, continuous time spans the run, and static time solves once."""
    assert DiscreteTime().fit().run(2.5).tolist() == [0.0, 1.0, 2.0]
    assert ContinuousTime().fit().run(3.0).tolist() == [0.0, 3.0]
    assert StaticTime().fit().run(3.0).tolist() == [0.0]
