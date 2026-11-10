"""Test the `_pulse` module of `skcomplex.systems.engineering.common.signals`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.engineering.common.signals import PulseInteraction


class DecayElement(Element):
    """Amount `c` that decays at rate 1."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c

    def _get_laws(self, s):
        return [RateLaw(s.c, -s.c)]


def _run(pulse, until):
    """Return the trajectory of a decaying amount with pulses."""
    system = CompositeSystem(elements=[DecayElement('body')], interactions=[pulse])
    return simulate(system, until=until, dt=0.25)


def _dose(**arguments):
    """Return pulses that add one unit to the amount."""
    return PulseInteraction('dose', target='body', reset={'c': lambda s: s.target.c + 1.0}, **arguments)


def test_pulses_fire_at_the_given_times():
    """Test that pulses at given times add their unit at exactly those times."""
    trajectory = _run(_dose(times=(1.0, 2.5)), 4.0)
    assert [event.time for event in trajectory.events] == pytest.approx([1.0, 2.5], abs=1e-6)
    expected = np.exp(-(4.0 - 2.5)) * (1 + np.exp(-1.5))
    assert trajectory.get_values('body.c')[-1] == pytest.approx(expected, rel=1e-3)


def test_periodic_pulses_fire_a_given_number_of_times():
    """Test that three periodic pulses from time 1 with period 2 fire at 1, 3, and 5, and then stop."""
    trajectory = _run(_dose(start=1.0, period=2.0, count=3), 10.0)
    assert [event.time for event in trajectory.events] == pytest.approx([1.0, 3.0, 5.0], abs=1e-6)
    assert all(event.label == 'dose' for event in trajectory.events)


def test_periodic_pulses_without_a_count_fire_until_the_end():
    """Test that periodic pulses without a count fire at every period after the initial time."""
    trajectory = _run(_dose(period=1.5), 7.0)
    assert [event.time for event in trajectory.events] == pytest.approx([1.5, 3.0, 4.5, 6.0], abs=1e-6)


def test_pulses_on_a_group_set_every_member_from_its_own_state():
    """Test that pulses on a group set the state of every member from that member's own state."""
    bodies = GroupElement('bodies', element=DecayElement, size=3, values={'c': [0.0, 1.0, 2.0]})
    dose = PulseInteraction('dose', target='bodies', reset={'c': lambda s: s.target.c + 1.0}, times=(1.0,))
    trajectory = simulate(CompositeSystem(elements=[bodies], interactions=[dose]), until=2.0, dt=0.25)
    expected = (np.array([0.0, 1.0, 2.0]) * np.exp(-1.0) + 1.0) * np.exp(-1.0)
    np.testing.assert_allclose(trajectory.get_values('bodies.c')[-1], expected, rtol=1e-3)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'reset': {}, 'times': (1.0,)}, 'Map state names'),
        ({'reset': {'c': 1.0}, 'times': (1.0,)}, 'Map state names'),
        ({'reset': {'c': lambda s: 0.0}}, 'have no time'),
        ({'reset': {'c': lambda s: 0.0}, 'period': 0.0}, 'positive number'),
        ({'reset': {'c': lambda s: 0.0}, 'period': 1.0, 'count': 0}, 'positive integer'),
    ],
)
def test_check_rejects_resets_and_times_it_cannot_take(arguments, match):
    """Test that pulses without states, without times, or with a bad period or count raise errors."""
    with pytest.raises(ValidationError, match=match):
        PulseInteraction('dose', target='body', **arguments).prepare()
