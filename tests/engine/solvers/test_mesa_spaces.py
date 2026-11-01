"""Test phases and the schedule of phases of a discrete time."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.system import System
from skcomplex.core.times import DiscreteTime
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import AgentView
from skcomplex.errors import SimulationError, ValidationError
from skcomplex.specs import PhaseSpec, PhasesScheduleSpec, ScheduleSpec, StateSpec, VisualSpec


def test_ordered_phases_are_a_schedule():
    """Test that a discrete time with phases lists them in order, and rejects repeated names."""
    time = (
        DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('produce'), PhaseSpec('trade'), PhaseSpec('consume')]),
        )
        .check()
        .prepare()
    )
    assert isinstance(time.schedule_, ScheduleSpec)
    assert [phase.name for phase in time.schedule_.phases] == ['produce', 'trade', 'consume']
    with pytest.raises(ValidationError, match='repeat a name'):
        DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('trade'), PhaseSpec('trade')])).check().prepare()


def test_phase_rejects_unknown_activation():
    """Test that a phase rejects an activation policy it does not know."""
    with pytest.raises(ValidationError, match='Unknown activation'):
        PhasesScheduleSpec([PhaseSpec('trade', activation='parallel')]).check()


class EmployerElement(Element):
    """Test helper Employer."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        offer: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.offer = offer

    def _get_laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.post, phase='offers')]

    def post(self, agent, model):
        """Test helper post."""
        agent.offer = model.time + 1.0


class WorkerElement(Element):
    """Test helper Worker."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        accepted: Annotated[float, StateSpec()] = -1.0,
        stale: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.accepted = accepted
        self.stale = stale

    def _get_laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.accept, phase='acceptance')]

    def accept(self, agent, model):
        """Test helper accept."""
        best = max(employer.offer for employer in model.agents('employers', AgentView))
        agent.stale += float(best != model.time + 1.0)
        agent.accepted = best


def test_offer_before_acceptance():
    """Every acceptance refers to an offer made earlier in the same step."""
    system = System(
        elements=[
            GroupElement('workers', element=WorkerElement, size=5),
            GroupElement('employers', element=EmployerElement, size=3),
        ],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('offers'), PhaseSpec('acceptance')])),
    )
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=10).trajectory_
    workers = trajectory.agents('workers', 10)
    assert np.all(workers.values['stale'] == 0)
    assert np.all(workers.values['accepted'] == trajectory.agents('employers', 10).values['offer'].max())


def test_undeclared_phase_raises():
    """A rule in a phase that the system does not declare raises an error."""
    system = System(elements=[GroupElement('employers', element=EmployerElement, size=2)])
    with pytest.raises(SimulationError, match='offers'):
        Simulation(system).check().prepare().run(until=1)


def test_phases_keep_unphased_models():
    """Rules without a phase run in the default phase."""

    class CounterElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            count: Annotated[float, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.count = count

        def _get_laws(self, s):
            return [RuleLaw(self.tick)]

        def tick(self, agent, model):
            agent.count += 1

    system = System(
        elements=[GroupElement('counters', element=CounterElement, size=3)],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('extra', repeat=2)])),
    )
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=4).trajectory_
    assert trajectory.agents('counters', 4).values['count'].tolist() == [4.0, 4.0, 4.0]


def test_sequential_and_by_type_order():
    """Sequential activation follows creation, and by-type activation follows the listed groups."""
    seen = []

    class FirstElement(Element):
        def _get_laws(self, s):
            return [RuleLaw(lambda agent, model: seen.append(('first', agent.id)), phase='go')]

    class SecondElement(Element):
        def _get_laws(self, s):
            return [RuleLaw(lambda agent, model: seen.append(('second', agent.id)), phase='go')]

    system = System(
        elements=[
            GroupElement('first', element=FirstElement, size=3),
            GroupElement('second', element=SecondElement, size=3),
        ],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('go', activation='by-type', groups=['second', 'first'])]),
        ),
    )
    Simulation(system, random_state=0).check().prepare().run(until=1)
    assert [group for group, _ in seen] == ['second'] * 3 + ['first'] * 3
    seen.clear()
    sequential = System(
        elements=[
            GroupElement('first', element=FirstElement, size=3),
            GroupElement('second', element=SecondElement, size=3),
        ],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('go', 'sequential')])),
    )
    Simulation(sequential, random_state=0).check().prepare().run(until=1)
    assert [identity for _, identity in seen] == sorted(identity for _, identity in seen)


def test_phase_checks_arguments():
    """A schedule rejects a phase with an unknown activation or fewer than one round."""
    with pytest.raises(ValidationError, match='activation'):
        PhasesScheduleSpec([PhaseSpec('x', activation='parallel')]).check()
    with pytest.raises(ValidationError, match='round'):
        PhasesScheduleSpec([PhaseSpec('x', repeat=0)]).check()
