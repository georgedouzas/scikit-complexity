"""Test phases and the schedule of phases of a discrete time."""

import numpy as np
import pytest

from skcomplex.core.base import SimulationError, ValidationError, state
from skcomplex.core.components import ElementComponent
from skcomplex.core.space import DiscreteTime, Schedule
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.agents import Phase, PhasesSchedule, RuleLaw


def test_ordered_phases_are_a_schedule():
    """Test that a discrete time with phases lists them in order, and rejects repeated names."""
    time = DiscreteTime(schedule=PhasesSchedule([Phase('produce'), Phase('trade'), Phase('consume')]))
    assert isinstance(time.schedule, Schedule)
    assert [phase.name for phase in time.schedule.phases] == ['produce', 'trade', 'consume']
    with pytest.raises(ValidationError, match='repeat a name'):
        PhasesSchedule([Phase('trade'), Phase('trade')])


def test_phase_rejects_unknown_activation():
    """Test that a phase rejects an activation policy it does not know."""
    with pytest.raises(ValueError, match='Unknown activation'):
        Phase('trade', activation='parallel')


class EmployerComponent(ElementComponent):
    """Test helper Employer."""

    offer: float = state(0.0)

    def laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.post, phase='offers')]

    def post(self, agent, model):
        """Test helper post."""
        agent.offer = model.time + 1.0


class WorkerComponent(ElementComponent):
    """Test helper Worker."""

    accepted: float = state(-1.0)
    stale: float = state(0.0)

    def laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.accept, phase='acceptance')]

    def accept(self, agent, model):
        """Test helper accept."""
        best = max(employer.offer for employer in model.agents('employers'))
        agent.stale += float(best != model.time + 1.0)
        agent.accepted = best


def test_offer_before_acceptance():
    """Every acceptance refers to an offer made earlier in the same step."""
    system = System(
        components=[WorkerComponent.group('workers', 5), EmployerComponent.group('employers', 3)],
        time=DiscreteTime(schedule=PhasesSchedule([Phase('offers'), Phase('acceptance')])),
    )
    trajectory = Simulation(system, random_state=0).run(until=10).trajectory_
    workers = trajectory.agents('workers', 10)
    assert np.all(workers.values['stale'] == 0)
    assert np.all(workers.values['accepted'] == trajectory.agents('employers', 10).values['offer'].max())


def test_undeclared_phase_raises():
    """A rule in a phase that the system does not declare raises an error."""
    system = System(components=[EmployerComponent.group('employers', 2)])
    with pytest.raises(SimulationError, match='offers'):
        Simulation(system).run(until=1)


def test_phases_keep_unphased_models():
    """Rules without a phase run in the default phase."""

    class CounterComponent(ElementComponent):
        count: float = state(0.0)

        def laws(self, s):
            return [RuleLaw(self.tick)]

        def tick(self, agent, model):
            agent.count += 1

    system = System(
        components=[CounterComponent.group('counters', 3)],
        time=DiscreteTime(schedule=PhasesSchedule([Phase('extra', repeat=2)])),
    )
    trajectory = Simulation(system, random_state=0).run(until=4).trajectory_
    assert trajectory.agents('counters', 4).values['count'].tolist() == [4.0, 4.0, 4.0]


def test_sequential_and_by_type_order():
    """Sequential activation follows creation, and by-type activation follows the listed groups."""
    seen = []

    class FirstComponent(ElementComponent):
        def laws(self, s):
            return [RuleLaw(lambda agent, model: seen.append(('first', agent.id)), phase='go')]

    class SecondComponent(ElementComponent):
        def laws(self, s):
            return [RuleLaw(lambda agent, model: seen.append(('second', agent.id)), phase='go')]

    system = System(
        components=[FirstComponent.group('first', 3), SecondComponent.group('second', 3)],
        time=DiscreteTime(schedule=PhasesSchedule([Phase('go', activation='by-type', groups=['second', 'first'])])),
    )
    Simulation(system, random_state=0).run(until=1)
    assert [group for group, _ in seen] == ['second'] * 3 + ['first'] * 3
    seen.clear()
    sequential = System(
        components=[FirstComponent.group('first', 3), SecondComponent.group('second', 3)],
        time=DiscreteTime(schedule=PhasesSchedule([Phase('go', 'sequential')])),
    )
    Simulation(sequential, random_state=0).run(until=1)
    assert [identity for _, identity in seen] == sorted(identity for _, identity in seen)


def test_phase_checks_arguments():
    """A phase rejects an unknown activation and fewer than one round."""
    with pytest.raises(ValueError, match='activation'):
        Phase('x', activation='parallel')
    with pytest.raises(ValueError, match='round'):
        Phase('x', repeat=0)
