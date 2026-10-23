"""Test phases and the schedule of phases of a discrete time."""

import pytest

from skcomplex.core.base import ValidationError
from skcomplex.core.formalisms.agents import Phase, Phases
from skcomplex.core.space import DiscreteTime, Schedule


def test_ordered_phases_are_a_schedule():
    """Test that a discrete time with phases lists them in order, and rejects repeated names."""
    time = DiscreteTime(schedule=Phases([Phase('produce'), Phase('trade'), Phase('consume')]))
    assert isinstance(time.schedule, Schedule)
    assert [phase.name for phase in time.schedule.phases] == ['produce', 'trade', 'consume']
    with pytest.raises(ValidationError, match='repeat a name'):
        Phases([Phase('trade'), Phase('trade')])


def test_phase_rejects_unknown_activation():
    """Test that a phase rejects an activation policy it does not know."""
    with pytest.raises(ValueError, match='Unknown activation'):
        Phase('trade', activation='parallel')
