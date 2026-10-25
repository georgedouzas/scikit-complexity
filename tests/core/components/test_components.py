"""Test the `_components` module of `skcomplex.core.components`."""

import pickle  # nosec B403: the tests pickle and load only objects they create
from typing import Any, ClassVar

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import (
    BaseInteraction,
    LabelError,
    NumericDomain,
    RoleError,
    ValidationError,
    VariableKind,
    VisualSpec,
    declarations_of,
    parameter,
    role,
    state,
)
from skcomplex.core.components import ElementComponent, GroupComponent
from skcomplex.core.laws import BoundaryLaw, JumpLaw, RateLaw
from skcomplex.core.space import ContinuousSpace, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.agents import RuleLaw
from skcomplex.formalisms.particles import ParticleComponent
from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationComponent, PredationInteraction

StockComponent = ElementComponent.define(
    'StockComponent',
    states={'n': state(1.0, domain=NumericDomain.NONNEGATIVE)},
    parameters={'k': 0.0},
)


GrowthInteraction = BaseInteraction.define(
    'GrowthInteraction',
    roles=['target'],
    parameters={'r': parameter(0.5, domain=NumericDomain.POSITIVE)},
    laws=lambda s: [RateLaw(s.target.n, s.r * s.target.n)],
)


def _decay_laws(self, s):
    """Return the decay of the stock at its own rate `k`."""
    return [RateLaw(s.n, -s.k * s.n)]


DecayingComponent = ElementComponent.define(
    'DecayingComponent',
    states={'n': 1.0},
    parameters={'k': 0.2},
    laws=_decay_laws,
)


class StockClassComponent(ElementComponent):
    """Stock written as a class."""

    n: float = state(1.0, domain=NumericDomain.NONNEGATIVE)
    k: float = parameter(0.0)


class GrowthClassInteraction(BaseInteraction):
    """Growth written as a class."""

    target: str = role()
    r: float = parameter(0.5, domain=NumericDomain.POSITIVE)

    def laws(self, s):
        """Return exponential growth."""
        return [RateLaw(s.target.n, s.r * s.target.n)]


class AttractionInteraction(BaseInteraction):
    """Pair interaction that returns a boundary law, which pair interactions do not take."""

    over: ClassVar[str] = 'pairs'
    position: ClassVar[str] = 'x'
    first: str = role()
    second: str = role()
    cutoff: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)

    def __init__(
        self,
        label: str,
        *,
        period: float | None = None,
        **values: Any,
    ) -> None:
        """Store the settings and pass the values on."""
        super().__init__(label, **values)
        self.period = period

    def laws(self, s):
        """Return a boundary law."""
        return [BoundaryLaw(s.first.v, s.first.v)]


BoxComponent = ElementComponent.define('BoxComponent', states={'n': 0.0})


TransferInteraction = BaseInteraction.define(
    'TransferInteraction',
    roles=['source', 'sink'],
    laws=lambda s: [RateLaw(s.sink.n, s.source.n)],
)


CellComponent = ElementComponent.define('CellComponent', states={'size': 0.0})


def _numpy_vector_rate(s):
    """Return a rate of a vector state written with a NumPy scalar factor."""
    return [RateLaw(s.v, np.float64(2.0) * s.v * s.v)]


def _numpy_decay(s):
    """Return a decay and a constant inflow written with NumPy scalars."""
    return [RateLaw(s.x, np.float64(-0.5) * s.x), RateLaw(s.x, np.float64(1.0))]


def _float_decay(s):
    """Return a decay and a constant inflow written with Python floats."""
    return [RateLaw(s.x, -0.5 * s.x), RateLaw(s.x, 1.0)]


def _parameter_jump(s):
    """Return a jump whose increment is a parameter."""
    return [JumpLaw(s.n, {s.n: s.k})]


def test_defined_types_match_classes():
    """Test that a defined model simulates exactly like the same model written with classes."""
    defined = System(components=[StockComponent('a', n=2.0)], interactions=[GrowthInteraction('g', target='a', r=0.3)])
    written = System(
        components=[StockClassComponent('a', n=2.0)],
        interactions=[GrowthClassInteraction('g', target='a', r=0.3)],
    )
    first = Simulation(defined, dt=0.5).run(until=2).trajectory_.states
    second = Simulation(written, dt=0.5).run(until=2).trajectory_.states
    assert np.array_equal(first, second)


def test_defined_types_declare_fields():
    """Test that defined types declare their states, parameters, and roles like classes."""
    assert set(declarations_of(StockComponent, VariableKind.STATE)) == {'n'}
    assert declarations_of(StockComponent, VariableKind.STATE)['n'].domain is NumericDomain.NONNEGATIVE
    assert set(declarations_of(StockComponent, VariableKind.PARAMETER)) == {'k'}
    assert set(declarations_of(GrowthInteraction, VariableKind.ROLE)) == {'target'}
    assert issubclass(StockComponent, ElementComponent)
    assert issubclass(GrowthInteraction, BaseInteraction)


def test_defined_types_form_groups():
    """Test that a group of 100 members of a defined type compiles once and exposes its parameters."""
    system = System(
        components=[StockComponent.group('herd', 100, n=np.linspace(1, 2, 100))],
        interactions=[GrowthInteraction('g', target='herd')],
    )
    trajectory = Simulation(system, dt=1).run(until=1).trajectory_
    assert trajectory['herd.n'].shape == (2, 100)
    assert np.allclose(trajectory['herd.n'][-1], np.linspace(1, 2, 100) * np.exp(0.5), rtol=1e-3)
    assert GrowthInteraction('g', target='herd', r=0.7).get_params() == {
        'label': 'g',
        'layer': None,
        'r': 0.7,
        'target': 'herd',
    }


def test_defined_types_clone_and_pickle():
    """Test that instances of defined types clone and pickle with their values."""
    growth = GrowthInteraction('g', target='a', r=0.7)
    assert clone(growth).get_params() == growth.get_params()
    data = pickle.dumps(System(components=[StockComponent('a', n=3.0)], interactions=[growth]))
    restored = pickle.loads(data)  # noqa: S301  # nosec B301: loads data the test just dumped
    assert restored.elements[0].n == 3.0
    assert restored.interactions[0].r == 0.7


def test_laws_function_receives_component():
    """Test that a laws function with two arguments receives the component and the namespace."""
    trajectory = Simulation(System(components=[DecayingComponent('a', n=1.0)]), dt=1).run(until=1).trajectory_
    assert np.isclose(trajectory['a.n'][-1], np.exp(-0.2), rtol=1e-3)


def test_group_stores_arguments():
    """Test that a group stores its label, size, and values unmodified."""
    positions = np.zeros((1000, 2))
    group = ParticleComponent.group('gas', 1000, x=positions)
    assert group.label == 'gas'
    assert group.size == 1000
    assert group.values['x'] is positions
    assert clone(group).size == 1000


def test_group_position_state_has_member_shape():
    """Test that a group's vector state has one row per member."""
    system = System(
        components=[ParticleComponent.group('gas', 1000, x=np.zeros((1000, 2)))],
        space=ContinuousSpace(n_dim=2),
    )
    assert compile_system(system).states.entry('gas.x').shape == (1000, 2)


def test_group_wrong_number_of_values_raises_error():
    """Test that checking a group with the wrong number of values names the group, the variable, and both sizes."""
    group = ParticleComponent.group('gas', 1000, x=np.zeros((999, 2)))
    with pytest.raises(ValidationError, match=r"'gas' has 999 values of x for 1000 members"):
        group.check()


def test_group_shared_and_lattice_values_are_accepted():
    """Test that shared values, per-member arrays, and lattice-shaped arrays pass the checks of group creation."""
    assert ParticleComponent.group('gas', 3, x=[0.5, 0.5]).size == 3
    assert PopulationComponent.group('herds', 3, n=[1.0, 2.0, 3.0]).size == 3
    assert (
        GroupComponent(
            'cells',
            element=PopulationComponent,
            space=LatticeSpace((2, 3)),
            on='sites',
            values={'n': np.ones((2, 3))},
        ).size
        == 6
    )


@pytest.mark.parametrize(('size', 'match'), [(0, 'positive integer'), (2.5, 'positive integer')])
def test_group_invalid_size_raises_error(size, match):
    """Test that checking a group rejects a size that is not a positive integer."""
    group = PopulationComponent.group('herds', size)
    with pytest.raises(ValidationError, match=match):
        group.check()


def test_group_unknown_value_raises_error():
    """Test that checking a group with an undeclared value names it and lists the declared states and parameters."""
    message = r'sets foo, which PopulationComponent does not declare\. Its states are n'
    group = PopulationComponent.group('herds', 3, foo=1)
    with pytest.raises(ValidationError, match=message):
        group.check()


def test_group_non_element_type_raises_error():
    """Test that checking a group rejects an element that is not an Element subclass."""
    group = GroupComponent('herds', element=int, size=3)
    with pytest.raises(ValidationError, match='not an Element subclass'):
        group.check()


def test_group_with_invalid_values_fails_validation_with_its_path():
    """Test that a group built with an invalid size is reported by compiling the system, with the path of the group."""
    system = System(components=[System('meadow', components=[PopulationComponent.group('herds', 0)])])
    with pytest.raises(ValidationError, match=r'meadow/herds: Group .herds. has size 0'):
        compile_system(system)


def test_group_default_visual_is_fitted():
    """Test that a fitted group holds the visual of its element type as its default visual."""
    group = PopulationComponent.group('herds', 3).fit()
    assert group.default_visual_ == PopulationComponent.visual
    assert list(group.declarations_) == []


def test_role_binds_to_group_and_applies_to_every_member():
    """Test that an interaction bound to a group applies its law to every member."""
    system = System(
        components=[PopulationComponent.group('herds', 3, n=[1.0, 2.0, 3.0])],
        interactions=[ExponentialGrowthInteraction('growth', target='herds', r=2.0)],
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [2.0, 4.0, 6.0]


def test_single_element_bound_with_group_receives_sum():
    """Test that a single element bound next to a group receives the sum over the members."""
    system = System(
        components=[PopulationComponent.group('rabbits', 3, n=[1.0, 2.0, 3.0]), PopulationComponent('fox', n=2.0)],
        interactions=[PredationInteraction('hunt', predator='fox', prey='rabbits', a=1.0, b=1.0)],
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [-2.0, -4.0, -6.0, 12.0]
    assert problem.has_group_sums
    assert 'fox.n' not in problem.equations


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_interaction_reads_two_groups_member_by_member(backend):
    """Test that an interaction bound to two groups of equal size reads row `i` of both together."""
    system = System(
        components=[BoxComponent.group('source', 5, n=np.arange(5.0)), BoxComponent.group('sink', 5)],
        interactions=[TransferInteraction('flow', source='source', sink='sink')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.backend == backend
    assert problem.states.read(drift, 'sink.n').tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_interaction_reading_groups_of_different_sizes_raises_error():
    """Test that an interaction reading groups of different sizes member by member names both groups and sizes."""
    system = System(
        components=[BoxComponent.group('source', 5), BoxComponent.group('sink', 7)],
        interactions=[TransferInteraction('flow', source='source', sink='sink')],
    )
    with pytest.raises(RoleError, match=r"sink of 7 members, source of 5 members.*over 'pairs'"):
        compile_system(system)


def test_group_takes_state_named_size():
    """Test that a state named size takes per-member values, since the group size is given by position."""
    group = CellComponent.group('cells', 3, size=[1.0, 2.0, 3.0])
    problem = compile_system(System(components=[group]))
    assert group.size == 3
    assert problem.states.read(problem.initial_state, 'cells.size').tolist() == [1.0, 2.0, 3.0]


def test_group_label_and_size_are_positional_only():
    """Test that the label and the size of a group cannot be given by keyword."""
    with pytest.raises(TypeError, match='positional argument'):
        BoxComponent.group(label='boxes', size=3)


def test_pair_interaction_rejects_other_law_kinds():
    """Test that pair interactions take rate laws only."""
    system = System(
        components=[ParticleComponent.group('gas', 3)],
        interactions=[AttractionInteraction('pull', first='gas', second='gas')],
        space=ContinuousSpace(n_dim=2),
    )
    with pytest.raises(ValidationError, match='laws other than Rate'):
        compile_system(system)


def test_pair_interaction_needs_vector_position():
    """Test that pair interactions need a vector position state."""

    class ContactInteraction(BaseInteraction):
        """Pair interaction between populations, which have no position."""

        over: ClassVar[str] = 'pairs'
        first: str = role()
        second: str = role()
        cutoff: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)

        def laws(self, s):
            """Return a rate law."""
            return [RateLaw(s.first.n, s.second.n)]

    system = System(
        components=[PopulationComponent.group('herds', 3)],
        interactions=[ContactInteraction('contact', first='herds', second='herds')],
    )
    with pytest.raises(ValidationError, match="vector state 'x'"):
        compile_system(system)


def test_lattice_keeps_visual_of_element_type():
    """Test that a lattice group of a type that declares a visual takes that visual, not an image."""
    ColumnComponent = ElementComponent.define('ColumnComponent', states={'h': 0.0}, visual=VisualSpec('heights'))
    assert ColumnComponent.lattice('columns', LatticeSpace((3,))).visual.kind == 'heights'


def test_lattice_without_type_visual_is_an_image():
    """Test that a lattice group of a type that declares no visual is drawn as an image."""
    CellComponent = ElementComponent.define('CellComponent', states={'h': 0.0})
    assert CellComponent.lattice('cells', LatticeSpace((3,))).visual.kind == 'image'


def test_element_stores_arguments():
    """Test that an element stores its label and initial state unmodified."""
    element = PopulationComponent('rabbits', n=40)
    assert element.label == 'rabbits'
    assert element.n == 40


def test_element_state_named_visual_raises_error():
    """Test that an element type with a state named visual names the state and lists the reserved names."""
    with pytest.raises(LabelError, match=r"'visual'.*label, remembers, visual are reserved"):
        ElementComponent.define('PaintedComponent', states={'visual': 0.0})


@pytest.mark.parametrize('name', ['location', 'alive', 'position'])
def test_agent_state_named_like_view_raises_error(name):
    """Test that an agent with a state named like an attribute of the agent view or of Mesa raises a label error."""
    SettlerComponent = ElementComponent.define(
        'SettlerComponent',
        states={name: 0.0},
        laws=lambda s: [RuleLaw(lambda agent, model: None)],
    )
    with pytest.raises(LabelError, match=rf'settlers declare the variables {name},.*agents_here.*reserved'):
        compile_system(System(components=[SettlerComponent.group('settlers', 2)]))


def test_numpy_scalar_times_vector_is_rejected():
    """Test that a law value that NumPy built as an array raises a named error that names the state."""
    MovingComponent = ElementComponent.define(
        'MovingComponent',
        states={'v': state(0.0, vector=True)},
        laws=_numpy_vector_rate,
    )
    system = System(components=[MovingComponent('m', v=[1.0, 2.0])], space=ContinuousSpace(n_dim=2))
    with pytest.raises(ValidationError, match=r'RateLaw of \(v_m.*float'):
        compile_system(system)


def test_numpy_scalar_values_match_floats():
    """Test that NumPy scalar law values run like the same laws written with Python floats."""
    trajectories = [
        Simulation(
            System(components=[ElementComponent.define('DecayComponent', states={'x': 2.0}, laws=laws)('d')]),
            dt=0.5,
        )
        .run(until=2)
        .trajectory_['d.x']
        for laws in (_numpy_decay, _float_decay)
    ]
    assert np.array_equal(*trajectories)


def test_jump_increment_symbol_is_rejected():
    """Test that a jump whose increment is a parameter raises a named error naming the state and the increment."""
    CounterComponent = ElementComponent.define(
        'CounterComponent',
        states={'n': 0.0},
        parameters={'k': 2.0},
        laws=_parameter_jump,
    )
    with pytest.raises(ValidationError, match=r'jump of n_c has the increment k_c.*number'):
        compile_system(System(components=[CounterComponent('c')]))
