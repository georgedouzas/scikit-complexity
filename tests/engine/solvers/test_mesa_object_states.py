"""Test the agent states of the `_mesa` module of `skcomplex.engine.solvers` that hold objects or have roles."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.analysis import AlignmentAnalysis
from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import ObservableSpec, StateSpec, VisualSpec


class QueueElement(Element):
    """Agent with a queue of pending orders that it appends to and pops from, each order an id."""

    reports = ('longest',)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        orders: Annotated[object, StateSpec(domain=None)] = (),
    ) -> None:
        super().__init__(label, visual=visual)
        self.orders = orders

    def _get_laws(self, s):
        """Return the rule of the queue."""
        return [RuleLaw(self._step)]

    def _step(self, agent, model):
        """Append the order of the step, and pop the oldest of a full queue."""
        orders = (*agent.orders, int(model.time))
        agent.orders = orders[1:] if len(orders) > 2 else orders
        model.report('longest', tuple(sorted(orders)))


def _queue_run(until=3):
    """Return the trajectory of two queues."""
    group = GroupElement('queues', element=QueueElement, size=2, values={'orders': [(), (7,)]})
    system = System(elements=[group], time=DiscreteTime())
    return Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_


def test_snapshots_keep_tuples_as_they_were():
    """Test that each snapshot keeps one tuple per agent, in a one-dimensional array of objects."""
    trajectory = _queue_run()
    first, last = trajectory.agents('queues', 0), trajectory.agents('queues', 3)
    assert first.values['orders'].shape == (2,)
    assert first.values['orders'].dtype == object
    assert first.values['orders'].tolist() == [(), (7,)]
    assert last.values['orders'].tolist() == [(2, 3), (2, 3)]


def test_composition_counts_agents_by_tuple():
    """Test that the composition of a group counts its agents by the tuples they hold."""
    seen = []

    class CountingElement(QueueElement):
        """Agent of a queue that records the composition of its group."""

        def _step(self, agent, model):
            """Record the composition, and then step."""
            seen.append(dict(model.composition('queues', 'orders')))
            super()._step(agent, model)

    group = GroupElement('queues', element=CountingElement, size=2, values={'orders': [(), (7,)]})
    Simulation(System(elements=[group], time=DiscreteTime()), random_state=0).check().prepare().run(until=1)
    assert seen[0] == {(): 1, (7,): 1}
    assert sum(seen[1].values()) == 2


def test_history_keeps_reported_objects():
    """Test that the reports of a run keep an object at every frame."""
    values = _queue_run().values('queues.longest')
    assert values.dtype == object
    assert isinstance(values[-1], tuple)


def test_series_analysis_refuses_a_state_of_objects_by_name():
    """Test that an analysis that needs numbers names the state that holds objects."""
    trajectory = _queue_run()
    with pytest.raises(ValidationError, match='The state orders of the agents of queues holds objects'):
        AlignmentAnalysis()._read_series(trajectory, ['queues.orders'])


class PhasedElement(Element):
    """Agent with a state named `phase` that is no heading, and a state with the role of heading."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phase: Annotated[float, StateSpec()] = 3.0,
        bearing: Annotated[float, StateSpec(role='heading')] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phase = phase
        self.bearing = bearing

    def _get_laws(self, s):
        """Return no change."""
        return [RuleLaw(lambda agent, model: None)]


def test_snapshots_read_headings_from_roles_alone():
    """Test that a state named `phase` stays a state, and the state with the role is the heading."""
    group = GroupElement('walkers', element=PhasedElement, size=3)
    trajectory = (
        Simulation(System(elements=[group], time=DiscreteTime()), random_state=0)
        .check()
        .prepare()
        .run(until=2)
        .trajectory_
    )
    snapshots = AlignmentAnalysis(group='walkers')._read_snapshots(trajectory, 'walkers')
    assert np.all(snapshots.headings == 0.5)
    assert np.all(snapshots.states['phase'] == 3.0)


def test_state_spec_rejects_an_unknown_role():
    """Test that a role is one of the names analyses read."""
    with pytest.raises(ValidationError, match='The role of a state'):
        StateSpec(role='spin').check()


def test_observable_keeps_a_hashable_value():
    """Test that an observable of a group by a function of its values may give an object."""
    observables = [ObservableSpec('lengths', group='queues', variable='orders', reduce=_lengths)]
    group = GroupElement('queues', element=QueueElement, size=2, values={'orders': [(), (7,)]})
    system = System(elements=[group], time=DiscreteTime())
    trajectory = Simulation(system, observables=observables, random_state=0).check().prepare().run(until=2).trajectory_
    assert trajectory.values('lengths')[0] == pytest.approx([0, 1])


def _lengths(values):
    """Return the length of the queue of every agent."""
    return np.asarray([len(item) for item in values], dtype=float)
