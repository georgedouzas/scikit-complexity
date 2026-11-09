"""Test the `_composite` module of `skcomplex.core.components.systems`."""

import importlib
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import CC, oo, var
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction, MapInteraction, RateInteraction
from skcomplex.core.components.laws import GeneratorLaw, RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import (
    BaseSpace,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    NetworkSpace,
    StaticTime,
)
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import UnitarySolver
from skcomplex.core.errors import RoleElementError, TagError, ValidationError
from skcomplex.core.specs import HilbertSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.epidemiology import SIRSystem


class BallElement(Element):
    """Point of a continuous space, with a vector state `x` of order 2."""

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, points at the values of `x`."""
        super()._prepare(context)
        self.default_visual_ = VisualSpec('points', {'position': 'x'})


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
    system = CompositeSystem(elements=[rabbits])
    assert (system.label, system.elements, system.interactions, system.systems) == ('system', [rabbits], (), ())
    assert (system.space, system.time) == (None, None)


def test_system_checks_every_part():
    """Test that checking a system checks its parts, which reject their invalid arguments."""
    with pytest.raises(ValidationError, match='label'):
        CompositeSystem(elements=[StockElement(1)]).prepare()
    with pytest.raises(ValidationError, match='Unknown schedule'):
        CompositeSystem(elements=[StockElement('a')], time=DiscreteTime('parallel')).prepare()


def test_system_prepares_groups_on_its_space():
    """Test that preparing a system prepares its space and gives it to the groups and nested systems without one."""
    weights = GroupElement('weights', element=StockElement, on='edges')
    nested = CompositeSystem('inner', elements=[GroupElement('nodes', element=StockElement, on='sites')])
    CompositeSystem(elements=[weights], systems=[nested], space=NetworkSpace(nx.cycle_graph(30))).prepare()
    assert weights.size_ == 30
    assert nested.elements[0].size_ == 30


def test_system_runs_into_the_laws_of_its_parts():
    """Test that a prepared system runs into the laws of its elements and its interactions, each for its namespace."""
    system = CompositeSystem(
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
    system = CompositeSystem('meadow', elements=[StockElement('rabbits')])
    assert repr(system) == "CompositeSystem(elements=[StockElement(label='rabbits')], label='meadow')"


def test_four_part_system_runs():
    """Test that a system of elements, interactions, no space, and continuous time runs and lists its parameters."""
    system = CompositeSystem(
        'ecology',
        elements=[StockElement('rabbits'), StockElement('foxes')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
        space=None,
        time=ContinuousTime(),
    )
    trajectory = simulate(system, until=1, dt=0.5)
    assert trajectory.get_values('rabbits.n')[-1] > 1.0
    assert simulate(system, until=0).problem.parameters.names == ['growth.r']


def test_systems_nest_and_interactions_couple_them_by_path():
    """Test that an interaction couples the elements of a nested system by their path."""
    system = CompositeSystem(
        systems=[CompositeSystem('meadow', elements=[StockElement('rabbits', n=2.0)])],
        interactions=[GrowthInteraction('growth', target='meadow/rabbits')],
    )
    trajectory = simulate(system, until=1, dt=0.5)
    assert trajectory.get_values('meadow/rabbits.n')[-1] > 2.0


def test_system_classes_nest_twice_under_their_labels():
    """Test that a ready-made system class nests twice by label, with its paths and its parameters kept apart."""
    system = CompositeSystem(
        systems=[SIRSystem(label='town'), CompositeSystem('region', systems=[SIRSystem(beta=0.0006, label='city')])],
        interactions=[RateInteraction('arrivals', target='town/S', state='x', rate=lambda s: 0.01 * s.target.x)],
    )
    assert PassCompiler().prepare(system).report_.ok
    assert system.prepare().system_.systems[0].elements[0].label == 'S'
    trajectory = simulate(system, until=10, dt=1.0)
    alone = simulate(SIRSystem(), until=10, dt=1.0)
    assert trajectory.get_values('town/S.x')[-1] > alone.get_values('S.x')[-1]
    assert trajectory.get_values('region/city/S.x')[-1] < alone.get_values('S.x')[-1]


def test_duplicate_labels_are_reported():
    """Test that two direct children with one label give one issue at the system that names the label."""
    report = PassCompiler().prepare(CompositeSystem('farm', elements=[StockElement('a'), StockElement('a')])).report_
    (issue,) = report.errors
    assert issue.path == 'farm'
    assert "'a'" in issue.message


def test_rate_under_discrete_time_is_reported():
    """Test that a rate law in a system of discrete time is reported with the time it needs."""
    system = CompositeSystem(
        elements=[StockElement('a')],
        interactions=[GrowthInteraction('g', target='a')],
        time=DiscreteTime(),
    )
    (issue,) = PassCompiler().prepare(system).report_.errors
    assert issue.code == 'law-time'
    assert 'needs ContinuousTime' in issue.message


def test_nested_system_inherits_time():
    """Test that a nested system without a time is checked against the time of its parent."""
    nested = CompositeSystem('inner', elements=[CounterElement('clock')])
    assert PassCompiler().prepare(CompositeSystem(systems=[nested], time=DiscreteTime())).report_.ok
    (issue,) = PassCompiler().prepare(CompositeSystem(systems=[nested], time=ContinuousTime())).report_.errors
    assert issue.path == 'inner/clock'


def test_static_time_rejects_dynamic_laws():
    """Test that a static system holding a rate law is reported."""
    system = CompositeSystem(
        elements=[StockElement('a')],
        interactions=[GrowthInteraction('g', target='a')],
        time=StaticTime(),
    )
    assert [issue.code for issue in PassCompiler().prepare(system).report_.errors] == ['law-time']


def test_periodic_space_wraps_positions():
    """Test that a particle crossing the side of a periodic space re-enters on the opposite side."""
    system = CompositeSystem(
        elements=[BallElement('p', x=[0.9, 0.5], x_d1=[1.0, 0.0])],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
    )
    trajectory = simulate(system, until=0.2, dt=0.2)
    assert trajectory.get_values('p.x')[-1].tolist() == pytest.approx([0.1, 0.5])


def test_periodic_box_wraps_member_with_same_velocity():
    """Test that a member leaving one side reappears at the opposite side with the same velocity."""
    system = CompositeSystem(
        elements=[
            GroupElement('gas', element=BallElement, size=1, values={'x': [[0.95, 0.5]], 'x_d1': [[1.0, 0.0]]}),
        ],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
    )
    trajectory = simulate(system, until=0.2, dt=0.1)
    assert np.allclose(trajectory.get_values('gas.x')[-1], [[0.15, 0.5]])
    assert np.allclose(trajectory.get_values('gas.x_d1')[-1], [[1.0, 0.0]])


def test_variable_group_of_elements_grows():
    """Test that a variable group of 10 dividing members has 20, 40, and 80 after one, two, and three steps."""
    system = CompositeSystem(elements=[GroupElement('cells', element=CellElement, size=10, variable=True)])
    trajectory = simulate(system, until=3)
    assert trajectory.get_values('cells.count').tolist() == [10.0, 20.0, 40.0, 80.0]


def test_group_on_edges_has_one_member_per_edge():
    """Test that a group on the edges of a network with 30 edges holds 30 members, sized from the inherited space."""
    weights = GroupElement('weights', element=StockElement, on='edges')
    problem = (
        PassCompiler()
        .prepare(CompositeSystem(elements=[weights], space=NetworkSpace(nx.cycle_graph(30))))
        .run()
        .problem_
    )
    assert problem.states.entry('weights.n').shape == (30,)


def test_group_on_edges_outside_a_network_is_reported():
    """Test that a group on edges in a lattice space is reported with the space it needs."""
    system = CompositeSystem(
        elements=[GroupElement('weights', element=StockElement, on='edges')],
        space=LatticeSpace(4),
    )
    report = PassCompiler().prepare(system).report_
    (issue,) = report.errors
    assert (issue.code, issue.message) == ('space-missing', 'weights: LatticeSpace has no edges.')


def test_element_with_hilbert_state_runs_unitary_dynamics():
    """Test that an element with a complex state in a Hilbert space and a generator law runs unitary dynamics."""
    system = CompositeSystem(elements=[TwoLevelElement('q')])
    trajectory = simulate(system, until=np.pi, solver=UnitarySolver(), dt=np.pi)
    assert trajectory.problem.form.name == 'OPERATOR'


class SwitchElement(Element):
    """Element with one boolean state."""

    def __init__(self, label: str, on: Annotated[int, StateSpec(FiniteEnumeratedSet([0, 1]))] = 0) -> None:
        super().__init__(label)
        self.on = on


def test_system_is_a_base_system_with_tags_derived_from_its_parts():
    """Test that a system of a rate and a noise interaction is stochastic, with both law kinds."""
    noisy = RateInteraction('growth', target='stock', state='n', rate=1.0, noise=0.1)
    system = CompositeSystem(elements=[StockElement('stock')], interactions=[noisy])
    tags = system.get_tags()
    assert isinstance(system, BaseSystem)
    assert (tags.laws, tags.randomness, tags.time, tags.space) == (
        {'rate', 'noise'},
        'stochastic',
        {'continuous'},
        {'none'},
    )
    assert tags.domains == {'real'}


def test_system_tags_reject_parts_on_spaces_that_nothing_joins():
    """Test that elements on a lattice and on a network, in a system without a space, disagree on the space."""
    on_lattice = GroupElement('cells', element=StockElement, space=LatticeSpace((3,)))
    on_network = GroupElement('nodes', element=StockElement, space=NetworkSpace(nx.cycle_graph(3)))
    with pytest.raises(TagError, match=r"The parts 'cells' and 'nodes' disagree on the tag 'space'"):
        CompositeSystem(elements=[on_lattice, on_network]).get_tags()
    joined = CompositeSystem(elements=[on_lattice, on_network], space=ContinuousSpace(n_dim=2)).get_tags()
    assert joined.space == {'continuous', 'lattice', 'network'}


def test_system_tags_reject_parts_of_times_that_nothing_joins():
    """Test that a rate and a map in one system disagree on the time, unless nested systems of each time join them."""
    rate = RateInteraction('growth', target='stock', state='n', rate=1.0)
    step = MapInteraction('step', target='counter/stock', state='n', function=lambda s: s.target.n)
    with pytest.raises(TagError, match=r"disagree on the tag 'time'"):
        CompositeSystem(elements=[StockElement('stock')], interactions=[rate, step]).get_tags()
    nested = CompositeSystem(
        systems=[
            CompositeSystem('plant', elements=[StockElement('stock')], time=ContinuousTime()),
            CompositeSystem('counter', elements=[StockElement('stock')], time=DiscreteTime()),
        ],
        interactions=[rate.set_params(target='plant/stock'), step],
    )
    assert nested.get_tags().time == {'continuous', 'discrete'}


def test_system_rejects_a_role_bound_to_a_domain_it_does_not_accept():
    """Test that a rate interaction on a boolean state fails to prepare, naming role, element, and domain."""
    rate = RateInteraction('growth', target='switch', state='on', rate=1.0)
    system = CompositeSystem(elements=[SwitchElement('switch')], interactions=[rate])
    with pytest.raises(
        RoleElementError,
        match=r"The role 'target' of 'growth' .* 'switch' .* 'on' of the domain 'boolean'",
    ):
        system.prepare()


@pytest.mark.parametrize(('name', 'module'), [('System', '_system'), ('MassActionSystem', '_mass_action')])
def test_former_class_names_are_gone(name, module):
    """Test that the former core system classes and their modules cannot be imported."""
    systems = importlib.import_module('skcomplex.core.components.systems')
    assert not hasattr(systems, name)
    with pytest.raises(ImportError):
        importlib.import_module(f'skcomplex.core.components.systems.{module}')
