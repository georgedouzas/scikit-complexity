"""Test the `_discrete_time` module of `skcomplex.core.components.spaces`."""

import pytest

from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, ScheduleSpec


def test_discrete_time_prepares_its_schedule():
    """Test that discrete time is synchronous by default and keeps a known schedule or a schedule spec."""
    assert DiscreteTime().prepare().schedule_ == 'synchronous'
    assert DiscreteTime('random-sequential').prepare().schedule_ == 'random-sequential'
    schedule = ScheduleSpec()
    assert DiscreteTime(schedule=schedule).prepare().schedule_ is schedule


def test_discrete_time_checks_its_schedule():
    """Test that an unknown schedule is stored as given and rejected by the check, and so is an invalid spec."""
    time = DiscreteTime(schedule='parallel')
    assert time.schedule == 'parallel'
    with pytest.raises(ValidationError, match='Unknown schedule'):
        time.prepare()
    with pytest.raises(ValidationError, match='repeat a name'):
        DiscreteTime(PhasesScheduleSpec((PhaseSpec('trade'), PhaseSpec('trade')))).prepare()


def test_discrete_time_steps_by_one():
    """Test that the frames of discrete time are the whole steps up to the end of the run."""
    assert DiscreteTime().prepare()._get_frames(2.5).tolist() == [0.0, 1.0, 2.0]
