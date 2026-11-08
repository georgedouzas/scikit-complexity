"""Test the `_crossing` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import CrossingInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
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
    system = System(elements=[LeakyElement('cell', **values)], interactions=[crossing])
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
    assert trajectory.values('cell.v').max() <= 1.0 + 1e-6


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
    trajectory = simulate(System(elements=[cells], interactions=[fire]), until=8.0, dt=0.05)
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
