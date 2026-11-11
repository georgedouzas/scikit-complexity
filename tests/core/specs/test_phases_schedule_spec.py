"""Test the `_phases_schedule_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, ScheduleSpec


@pytest.mark.parametrize(
    ('phases', 'message'),
    [
        ((PhaseSpec('trade'), PhaseSpec('trade')), 'repeat a name'),
        ((PhaseSpec('trade', activation='parallel'),), 'Unknown activation'),
        ((PhaseSpec('trade', repeat=0),), 'at least one round'),
        ((PhaseSpec('trade', activation='visits', visits=-1.0),), 'at least 0'),
        ((PhaseSpec('trade', period=0),), 'period of at least one step'),
    ],
)
def test_phases_schedule_spec_checks_its_phases(phases, message):
    """Test that phases need their own names, a known activation, and at least one round."""
    with pytest.raises(ValidationError, match=message):
        PhasesScheduleSpec(phases)._check()


def test_ordered_phases_are_a_schedule():
    """Test that a discrete time with phases lists them in order, and rejects repeated names."""
    time = DiscreteTime(
        schedule=PhasesScheduleSpec([PhaseSpec('produce'), PhaseSpec('trade'), PhaseSpec('consume')]),
    ).prepare()
    assert isinstance(time.schedule_, ScheduleSpec)
    assert [phase.name for phase in time.schedule_.phases] == ['produce', 'trade', 'consume']
    with pytest.raises(ValidationError, match='repeat a name'):
        DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('trade'), PhaseSpec('trade')])).prepare()


def test_phase_rejects_unknown_activation():
    """Test that a phase rejects an activation policy it does not know."""
    with pytest.raises(ValidationError, match='Unknown activation'):
        PhasesScheduleSpec([PhaseSpec('trade', activation='parallel')])._check()


def test_phase_checks_arguments():
    """A schedule rejects a phase with an unknown activation or fewer than one round."""
    with pytest.raises(ValidationError, match='activation'):
        PhasesScheduleSpec([PhaseSpec('x', activation='parallel')])._check()
    with pytest.raises(ValidationError, match='round'):
        PhasesScheduleSpec([PhaseSpec('x', repeat=0)])._check()
