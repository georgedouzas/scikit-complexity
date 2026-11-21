"""Test the `_discrete_time` module of `skcomplex.core.components.spaces`."""

import pytest

from skcomplex.core.components.spaces import ScheduleSpec
from skcomplex.core.components.spaces._discrete_time import _DiscreteTime
from skcomplex.core.engine.compile import PhaseSpec, PhasesScheduleSpec
from skcomplex.core.errors import ValidationError


def test_discrete_time_prepares_its_schedule():
    """Test that discrete time is synchronous by default and keeps a known schedule or a schedule spec."""
    assert _DiscreteTime().prepare().schedule_ == 'synchronous'
    assert _DiscreteTime('random-sequential').prepare().schedule_ == 'random-sequential'
    schedule = ScheduleSpec()
    assert _DiscreteTime(schedule=schedule).prepare().schedule_ is schedule


def test_discrete_time_checks_its_schedule():
    """Test that an unknown schedule is stored as given and rejected by the check, and so is an invalid spec."""
    time = _DiscreteTime(schedule='parallel')
    assert time.schedule == 'parallel'
    with pytest.raises(ValidationError, match='Unknown schedule'):
        time.prepare()
    with pytest.raises(ValidationError, match='repeat a name'):
        _DiscreteTime(PhasesScheduleSpec((PhaseSpec('trade'), PhaseSpec('trade')))).prepare()
