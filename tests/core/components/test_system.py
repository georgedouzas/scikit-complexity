"""Test the `system` module of `skcomplex.core.components`."""

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import CC, oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement, PointElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import GeneratorLaw, RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import (
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    NetworkSpace,
    StaticTime,
)
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import OperatorCompiler, PassCompiler, compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import UnitarySolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import HilbertSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec

PAULI = HilbertSpec(
    2,
    {'x': (np.array([[0.0, 1.0], [1.0, 0.0]]), True), 'z': (np.diag([1.0, -1.0]), True)},
    basis={'up': 0, 'down': 1},
)


class StockElement(Element):
    """Element with one nonnegative state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


class GrowthInteraction(Interaction):
    """Exponential growth of the state of its target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s):
        """Return the rate of growth."""
        return [RateLaw(s.target.n, s.r * s.target.n)]


class CounterElement(Element):
    """Element whose state grows by one every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k

    def _get_laws(self, s):
        """Return the update of the count."""
        return [UpdateLaw(s.k, s.k + 1)]


class CellElement(Element):
    """Element whose members divide every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        size: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.size = size

    def _get_laws(self, s):
        """Return the rule of division."""
        return [RuleLaw(self.divide)]

    def divide(self, agent, system):
        """Create one offspring."""
        del system
        agent.create()


class TwoLevelElement(Element):
    """Element whose state lies in a two-dimensional Hilbert space, rotated about `x`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        hilbert: HilbertSpec = PAULI,
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert

    def _get_laws(self, s):
        """Return the generator of the rotation about `x`."""
        return [GeneratorLaw(local_operator(s.psi, 'x') / 2)]


def test_system_constructor_only_stores_its_parts():
    """Test that a system stores its parts as given and is labelled `system` by default."""
    rabbits = StockElement('rabbits')
    system = System(elements=[rabbits])
    assert (system.label, system.elements, system.interactions, system.systems) == ('system', [rabbits], (), ())
    assert (system.space, system.time) == (None, None)


def test_system_checks_every_part():
    """Test that checking a system checks its parts, which reject their invalid arguments."""
    with pytest.raises(ValidationError, match='label'):
        System(elements=[StockElement(1)]).prepare()
    with pytest.raises(ValidationError, match='Unknown schedule'):
        System(elements=[StockElement('a')], time=DiscreteTime('parallel')).prepare()


def test_system_prepares_groups_on_its_space():
    """Test that preparing a system prepares its space and gives it to the groups and nested systems without one."""
    weights = GroupElement('weights', element=StockElement, on='edges')
    nested = System('inner', elements=[GroupElement('nodes', element=StockElement, on='sites')])
    System(elements=[weights], systems=[nested], space=NetworkSpace(nx.cycle_graph(30))).prepare()
    assert weights.size_ == 30
    assert nested.elements[0].size_ == 30


def test_system_runs_into_the_laws_of_its_parts():
    """Test that a prepared system runs into the laws of its elements and its interactions, each for its namespace."""
    system = System(
        elements=[CounterElement('clock'), StockElement('rabbits')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
        time=ContinuousTime(),
    )
    n, r, k = var('n r k')
    rabbits = SimpleNamespace(n=n)
    namespace = SimpleNamespace(
        clock=SimpleNamespace(k=k),
        rabbits=rabbits,
        growth=SimpleNamespace(target=rabbits, r=r),
    )
    laws = system.prepare().run(namespace).laws_
    assert [type(law).__name__ for law in laws] == ['UpdateLaw', 'RateLaw']
    assert laws[1].prepare().run({'n': 2.0, 'r': 0.5}).value_ == 1.0


def test_system_prints_its_parameters():
    """Test that a system prints the parameters it was given, through scikit-learn."""
    system = System('meadow', elements=[StockElement('rabbits')])
    assert repr(system) == "System(elements=[StockElement(label='rabbits')], label='meadow')"


def test_four_part_system_runs():
    """Test that a system of elements, interactions, no space, and continuous time runs and lists its parameters."""
    system = System(
        'ecology',
        elements=[StockElement('rabbits'), StockElement('foxes')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
        space=None,
        time=ContinuousTime(),
    )
    trajectory = Simulation(system, dt=0.5).prepare().run(until=1).trajectory_
    assert trajectory.values('rabbits.n')[-1] > 1.0
    assert Simulation(system).prepare().run(until=0).problem_.parameters.names == ['growth.r']


def test_systems_nest_and_interactions_couple_them_by_path():
    """Test that an interaction couples the elements of a nested system by their path."""
    system = System(
        systems=[System('meadow', elements=[StockElement('rabbits', n=2.0)])],
        interactions=[GrowthInteraction('growth', target='meadow/rabbits')],
    )
    trajectory = Simulation(system, dt=0.5).prepare().run(until=1).trajectory_
    assert trajectory.values('meadow/rabbits.n')[-1] > 2.0


def test_duplicate_labels_are_reported():
    """Test that two direct children with one label give one issue at the system that names the label."""
    report = PassCompiler().prepare(System('farm', elements=[StockElement('a'), StockElement('a')])).report_
    (issue,) = report.errors
    assert issue.path == 'farm'
    assert "'a'" in issue.message


def test_rate_under_discrete_time_is_reported():
    """Test that a rate law in a system of discrete time is reported with the time it needs."""
    system = System(
        elements=[StockElement('a')],
        interactions=[GrowthInteraction('g', target='a')],
        time=DiscreteTime(),
    )
    (issue,) = PassCompiler().prepare(system).report_.errors
    assert issue.code == 'law-time'
    assert 'needs ContinuousTime' in issue.message


def test_nested_system_inherits_time():
    """Test that a nested system without a time is checked against the time of its parent."""
    nested = System('inner', elements=[CounterElement('clock')])
    assert PassCompiler().prepare(System(systems=[nested], time=DiscreteTime())).report_.ok
    (issue,) = PassCompiler().prepare(System(systems=[nested], time=ContinuousTime())).report_.errors
    assert issue.path == 'inner/clock'


def test_static_time_rejects_dynamic_laws():
    """Test that a static system holding a rate law is reported."""
    system = System(
        elements=[StockElement('a')],
        interactions=[GrowthInteraction('g', target='a')],
        time=StaticTime(),
    )
    assert [issue.code for issue in PassCompiler().prepare(system).report_.errors] == ['law-time']


def test_periodic_space_wraps_positions():
    """Test that a particle crossing the side of a periodic space re-enters on the opposite side."""
    system = System(
        elements=[PointElement('p', x=[0.9, 0.5], x_d1=[1.0, 0.0])],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
    )
    trajectory = Simulation(system, dt=0.2).prepare().run(until=0.2).trajectory_
    assert trajectory.values('p.x')[-1].tolist() == pytest.approx([0.1, 0.5])


def test_periodic_box_wraps_member_with_same_velocity():
    """Test that a member leaving one side reappears at the opposite side with the same velocity."""
    system = System(
        elements=[
            GroupElement('gas', element=PointElement, size=1, values={'x': [[0.95, 0.5]], 'x_d1': [[1.0, 0.0]]}),
        ],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
    )
    trajectory = Simulation(system, dt=0.1).prepare().run(until=0.2).trajectory_
    assert np.allclose(trajectory.values('gas.x')[-1], [[0.15, 0.5]])
    assert np.allclose(trajectory.values('gas.x_d1')[-1], [[1.0, 0.0]])


def test_variable_group_of_elements_grows():
    """Test that a variable group of 10 dividing members has 20, 40, and 80 after one, two, and three steps."""
    system = System(elements=[GroupElement('cells', element=CellElement, size=10, variable=True)])
    trajectory = Simulation(system).prepare().run(until=3).trajectory_
    assert trajectory.values('cells.count').tolist() == [10.0, 20.0, 40.0, 80.0]


def test_group_on_edges_has_one_member_per_edge():
    """Test that a group on the edges of a network with 30 edges holds 30 members, sized from the inherited space."""
    weights = GroupElement('weights', element=StockElement, on='edges')
    problem = compile_system(System(elements=[weights], space=NetworkSpace(nx.cycle_graph(30))))
    assert problem.states.entry('weights.n').shape == (30,)


def test_group_on_edges_outside_a_network_is_reported():
    """Test that a group on edges in a lattice space is reported with the space it needs."""
    system = System(elements=[GroupElement('weights', element=StockElement, on='edges')], space=LatticeSpace(4))
    report = PassCompiler().prepare(system).report_
    (issue,) = report.errors
    assert (issue.code, issue.message) == ('space-missing', 'LatticeSpace has no edges.')


def test_element_with_hilbert_state_runs_unitary_dynamics():
    """Test that an element with a complex state in a Hilbert space and a generator law runs unitary dynamics."""
    system = System(elements=[TwoLevelElement('q')])
    simulation = Simulation(system, compiler=OperatorCompiler(), solver=UnitarySolver(), dt=np.pi).prepare()
    simulation.run(until=np.pi)
    assert simulation.problem_.form.name == 'OPERATOR'
