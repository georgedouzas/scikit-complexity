"""Test systems defined by their components, interactions, space, and time."""

import pytest

from skcomplex.core.base import Domain, ValidationError, parameter, role, state
from skcomplex.core.compile import validate
from skcomplex.core.model import (
    ContinuousTime,
    DiscreteTime,
    Element,
    Interaction,
    Phase,
    Phases,
    Rate,
    StaticTime,
    System,
    Update,
)
from skcomplex.core.runtime import Simulation


class Stock(Element):
    """Element with one nonnegative state."""

    n: float = state(1.0, domain=Domain.NONNEGATIVE)


class Growth(Interaction):
    """Exponential growth of the state of its target."""

    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        """Return the rate of growth."""
        return [Rate(s.target.n, s.r * s.target.n)]


class Counter(Element):
    """Element whose state grows by one every step."""

    k: float = state(0.0)

    def laws(self, s):
        """Return the update of the count."""
        return [Update(s.k, s.k + 1)]


def test_four_part_system_runs():
    """Test that a system of components, interactions, no space, and continuous time runs and lists its parameters."""
    system = System(
        'ecology',
        components=[Stock('rabbits'), Stock('foxes')],
        interactions=[Growth('growth', target='rabbits')],
        space=None,
        time=ContinuousTime(),
    )
    trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
    assert trajectory['rabbits.n'][-1] > 1.0
    assert Simulation(system).run(until=0).problem_.parameters.names == ['growth.r']


@pytest.mark.parametrize(
    ('argument', 'home'),
    [
        ('noise', 'Noise(..., convention=...)'),
        ('history', 'state(history=...)'),
        ('phases', 'DiscreteTime(schedule=Phases([...]))'),
        ('observables', 'Simulation(observables=...)'),
        ('elements', '`components`'),
    ],
)
def test_execution_arguments_are_rejected(argument, home):
    """Test that an argument of the former signature raises a type error that names its new home."""
    with pytest.raises(TypeError, match=argument) as error:
        System(**{argument: []})
    assert home in str(error.value)


def test_systems_nest_as_components():
    """Test that nested systems are components and an interaction couples their elements by path."""
    system = System(
        components=[System('meadow', components=[Stock('rabbits', n=2.0)])],
        interactions=[Growth('growth', target='meadow/rabbits')],
    )
    assert system.systems[0].label == 'meadow'
    assert system.elements == ()
    trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
    assert trajectory['meadow/rabbits.n'][-1] > 2.0


def test_duplicate_labels_are_reported():
    """Test that two direct children with one label give one issue at the system that names the label."""
    report = validate(System('farm', components=[Stock('a'), Stock('a')]))
    (issue,) = report.errors
    assert issue.path == 'farm'
    assert "'a'" in issue.message


def test_rate_under_discrete_time_is_reported():
    """Test that a rate law in a system of discrete time is reported with the time it needs."""
    system = System(components=[Stock('a')], interactions=[Growth('g', target='a')], time=DiscreteTime())
    (issue,) = validate(system).errors
    assert issue.code == 'law-time'
    assert 'continuous' in issue.message
    assert 'update' in issue.fix


def test_nested_system_inherits_time():
    """Test that a nested system without a time is checked against the time of its parent."""
    nested = System('inner', components=[Counter('clock')])
    assert validate(System(components=[nested], time=DiscreteTime())).ok
    (issue,) = validate(System(components=[nested], time=ContinuousTime())).errors
    assert issue.path == 'inner/clock'


def test_ordered_phases_are_a_schedule():
    """Test that a discrete time with phases lists them in order, and rejects repeated names."""
    time = DiscreteTime(schedule=Phases([Phase('produce'), Phase('trade'), Phase('consume')]))
    assert [phase.name for phase in time.phases] == ['produce', 'trade', 'consume']
    assert DiscreteTime().phases == ()
    with pytest.raises(ValidationError, match='repeat a name'):
        Phases([Phase('trade'), Phase('trade')])


def test_unknown_schedule_is_rejected():
    """Test that a schedule other than the known ones or phases is rejected."""
    with pytest.raises(ValidationError, match='schedule'):
        DiscreteTime(schedule='parallel')


def test_static_time_rejects_dynamic_laws():
    """Test that a static system holding a rate law is reported."""
    system = System(components=[Stock('a')], interactions=[Growth('g', target='a')], time=StaticTime())
    assert [issue.code for issue in validate(system).errors] == ['law-time']
