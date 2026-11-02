"""Test the `_event_locator` module of `skcomplex.engine.solvers`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import CrossingInteraction
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.specs import StateSpec, VisualSpec


class CounterElement(Element):
    """Element with a tally that resets change."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        tally: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.tally = tally


def _doubling() -> CrossingInteraction:
    """Return a crossing at time 0.5 that doubles the tally and adds 1."""
    return CrossingInteraction(
        'doubling',
        target='counter',
        guard=lambda n: n.t - 0.5,
        reset={'tally': lambda n: n.target.tally * 2 + 1},
    )


def _adding() -> CrossingInteraction:
    """Return a crossing at time 0.5 that adds 10 to the tally."""
    return CrossingInteraction(
        'adding',
        target='counter',
        guard=lambda n: n.t - 0.5,
        reset={'tally': lambda n: n.target.tally + 10},
    )


@pytest.mark.parametrize('solver', [None, RK4Solver(dt=0.01)])
def test_apply_runs_tied_crossings_in_declaration_order(solver):
    """Test that crossings at one time all fire, each reading the state the crossing before it left."""
    for crossings, expected, order in (
        ((_doubling(), _adding()), 11.0, ['doubling', 'adding']),
        ((_adding(), _doubling()), 21.0, ['adding', 'doubling']),
    ):
        system = System(elements=[CounterElement('counter')], interactions=list(crossings))
        trajectory = Simulation(system, solver=solver).check().prepare().run(until=1.0).trajectory_
        assert trajectory.values('counter.tally')[-1] == pytest.approx(expected)
        assert [event.label for event in trajectory.events] == order


def test_apply_fires_every_member_of_a_group_that_crosses_at_one_time():
    """Test that identical members whose guards cross at one time each fire their reset."""
    group = GroupElement('counter', element=CounterElement, size=3)
    crossing = CrossingInteraction(
        'tick',
        target='counter',
        guard=lambda n: n.t - 0.5 - n.target.tally,
        reset={'tally': lambda n: n.target.tally + 1},
    )
    system = System(elements=[group], interactions=[crossing])
    trajectory = Simulation(system).check().prepare().run(until=1.0).trajectory_
    assert np.allclose(trajectory.values('counter.tally')[-1], 1.0)
    assert sorted(event.member for event in trajectory.events) == [0, 1, 2]
