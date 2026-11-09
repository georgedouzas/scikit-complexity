"""Test the `_crossing` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import CrossingInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import periodic_guard
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


class LeakyElement(Element):
    """Leaky integrator `v' = (drive - v) / tau`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        drive: Annotated[float, ParameterSpec()] = 2.0,
        tau: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.drive = drive
        self.tau = tau

    def _get_laws(self, s):
        return [RateLaw(s.v, (s.drive - s.v) / s.tau)]


def _events(crossing, until, **values):
    """Return the times and labels of the events of a leaky integrator with a crossing."""
    system = CompositeSystem(elements=[LeakyElement('cell', **values)], interactions=[crossing])
    trajectory = simulate(system, until=until, dt=0.1)
    return trajectory, [event.time for event in trajectory.events]


def test_integrate_and_fire_fires_with_the_closed_form_period():
    """Test that a leaky integrator reset at a threshold fires with the period `tau log(I / (I - theta))`."""
    fire = CrossingInteraction(
        'fire',
        target='cell',
        guard=lambda s: s.target.v - 1.0,
        reset={'v': lambda s: 0.0},
        direction=1,
    )
    trajectory, times = _events(fire, 10.0, drive=2.0, tau=1.5)
    period = 1.5 * np.log(2.0 / (2.0 - 1.0))
    assert len(times) == int(10.0 // period)
    np.testing.assert_allclose(np.diff(times), period, rtol=1e-3)
    assert all(event.label == 'fire' for event in trajectory.events)
    assert trajectory.get_values('cell.v').max() <= 1.0 + 1e-6


def test_integrate_and_fire_group_fires_every_member_with_its_own_period():
    """Test that every member of a group of leaky integrators fires with its own period and resets alone."""
    drives = np.array([1.5, 2.0, 3.0])
    cells = GroupElement('cells', element=LeakyElement, size=3, values={'drive': drives})
    fire = CrossingInteraction(
        'fire',
        target='cells',
        guard=lambda s: s.target.v - 1.0,
        reset={'v': lambda s: 0.0},
        direction=1,
    )
    trajectory = simulate(CompositeSystem(elements=[cells], interactions=[fire]), until=8.0, dt=0.05)
    events = trajectory.events
    for member, drive in enumerate(drives):
        times = [event.time for event in events if event.member == member]
        np.testing.assert_allclose(np.diff(times), np.log(drive / (drive - 1.0)), rtol=1e-3)
    for event in events:
        changed = np.flatnonzero(event.before != event.after)
        assert changed.tolist() == [event.member]


def test_terminal_crossing_ends_the_run():
    """Test that a terminal crossing ends the run at the crossing."""
    stop = CrossingInteraction('stop', target='cell', guard=lambda s: s.target.v - 1.0, terminal=True)
    trajectory, times = _events(stop, 10.0)
    assert times == pytest.approx([np.log(2.0)], rel=1e-3)
    assert trajectory.times[-1] == pytest.approx(np.log(2.0), rel=1e-3)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'guard': lambda s: s.target.v, 'direction': 2}, 'Use -1, 0, or 1'),
        ({'guard': 1.0}, 'functions of the namespace'),
        ({'guard': lambda s: s.target.v, 'reset': {'v': 0.0}}, 'functions of the namespace'),
    ],
)
def test_check_rejects_directions_and_functions_it_cannot_take(arguments, match):
    """Test that an unknown direction, or a guard or a new value that is not a function, raise errors."""
    with pytest.raises(ValidationError, match=match):
        CrossingInteraction('crossing', target='cell', **arguments).prepare()


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


def test_periodic_impulses_by_a_guard():
    """Test that a periodic guard gives impulses at `start + j period`, `count` times."""
    dose = CrossingInteraction(
        'dose',
        target='body',
        guard=lambda s: periodic_guard(s.t, 2.0, start=1.0, count=3),
        reset={'c': lambda s: s.target.c + 1.0},
    )
    trajectory = simulate(CompositeSystem(elements=[DecayElement('body')], interactions=[dose]), until=10.0, dt=0.25)
    assert [event.time for event in trajectory.events] == pytest.approx([1.0, 3.0, 5.0], abs=1e-6)


def test_impulses_at_given_times():
    """Test that guards `t - t_k` give impulses at the given times."""
    doses = [
        CrossingInteraction(
            f'dose_{index}',
            target='body',
            guard=lambda s, time=time: s.t - time,
            reset={'c': lambda s: s.target.c + 1.0},
            direction=1,
        )
        for index, time in enumerate((1.0, 2.5))
    ]
    trajectory = simulate(CompositeSystem(elements=[DecayElement('body')], interactions=doses), until=4.0, dt=0.25)
    assert [event.time for event in trajectory.events] == pytest.approx([1.0, 2.5], abs=1e-6)
    expected = np.exp(-(4.0 - 2.5)) * (1 + np.exp(-1.5))
    assert trajectory.get_values('body.c')[-1] == pytest.approx(expected, rel=1e-3)


def test_a_guard_that_reads_no_member_sets_every_member_of_a_group_at_once():
    """Test that a time guard over a group sets every member from its own state at one event."""
    bodies = GroupElement('bodies', element=DecayElement, size=3, values={'c': [0.0, 1.0, 2.0]})
    dose = CrossingInteraction(
        'dose',
        target='bodies',
        guard=lambda s: s.t - 1.0,
        reset={'c': lambda s: s.target.c + 1.0},
        direction=1,
    )
    trajectory = simulate(CompositeSystem(elements=[bodies], interactions=[dose]), until=2.0, dt=0.25)
    assert len(trajectory.events) == 1
    expected = (np.array([0.0, 1.0, 2.0]) * np.exp(-1.0) + 1.0) * np.exp(-1.0)
    np.testing.assert_allclose(trajectory.get_values('bodies.c')[-1], expected, rtol=1e-3)
