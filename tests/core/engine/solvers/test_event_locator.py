"""Test the `_event_locator` module of `skcomplex.core.engine.solvers`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EventLaw
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.systems import System


def _crossing(label: str, guard, reset) -> Interaction:
    """Return an interaction of the counter whose crossing of a guard resets the tally."""
    return Interaction(
        label,
        roles={'target': 'counter'},
        laws=lambda s: [EventLaw(guard(s), {s.target.tally: reset(s)}, label=label)],
    )


def _doubling() -> Interaction:
    """Return a crossing at time 0.5 that doubles the tally and adds 1."""
    return _crossing('doubling', lambda n: n.t - 0.5, lambda n: n.target.tally * 2 + 1)


def _adding() -> Interaction:
    """Return a crossing at time 0.5 that adds 10 to the tally."""
    return _crossing('adding', lambda n: n.t - 0.5, lambda n: n.target.tally + 10)


@pytest.mark.parametrize('solver', ['auto', RK4Solver(dt=0.01)])
def test_apply_runs_tied_crossings_in_declaration_order(solver):
    """Test that crossings at one time all fire, each reading the state the crossing before it left."""
    for crossings, expected, order in (
        ((_doubling(), _adding()), 11.0, ['doubling', 'adding']),
        ((_adding(), _doubling()), 21.0, ['adding', 'doubling']),
    ):
        counter = Element('counter', values={'tally': 0.0})
        system = System(elements=[counter], interactions=list(crossings), solver=solver)
        trajectory = system.prepare().simulate(until=1.0)
        assert trajectory.get_values('counter.tally')[-1] == pytest.approx(expected)
        assert [event.label for event in trajectory.events] == order


def test_apply_fires_every_member_of_a_group_that_crosses_at_one_time():
    """Test that identical members whose guards cross at one time each fire their reset."""
    group = Element('counter', values={'tally': np.zeros(3)}, size=3)
    crossing = _crossing('tick', lambda n: n.t - 0.5 - n.target.tally, lambda n: n.target.tally + 1)
    trajectory = System(elements=[group], interactions=[crossing]).prepare().simulate(until=1.0)
    assert np.allclose(trajectory.get_values('counter.tally')[-1], 1.0)
    assert sorted(event.member for event in trajectory.events) == [0, 1, 2]
