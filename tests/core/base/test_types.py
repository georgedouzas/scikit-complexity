"""Test the `_types` module of `skcomplex.core.base`."""

import copy
import pickle  # nosec B403: the test loads only the pickles it makes

import numpy as np
import pytest

from skcomplex.core.base import (
    BaseInteraction,
    Form,
    LawKind,
    NoiseKind,
    NumericDomain,
    RegistryError,
    SpaceKind,
    Term,
    Trait,
    ValidationError,
    VariableKind,
    declarations_of,
    parameter,
    role,
    state,
)
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver, HeunSolver, MilsteinSolver

from .conftest import PersonComponent, ProgressionInteraction


class Colour(Term):
    """Vocabulary of colours, defined only for these tests."""


def test_term_registers_by_name():
    """Test that registering a registered name with its value returns a term equal to the registered one."""
    assert Form.register('ODE', 'ordinary differential equations') == Form.ODE


def test_term_rejects_other_value():
    """Test that a registered name with another value raises a named error that names both values."""
    with pytest.raises(RegistryError, match="'ordinary differential equations'"):
        Form.register('ODE', 'other equations')


def test_term_registers_new_terms_per_vocabulary():
    """Test that a new term joins its own vocabulary only, after the registered ones."""
    red = Colour.register('RED', 'red')
    assert Colour.registered()[-1] == red
    assert red not in Form.registered()
    assert Colour.register('RED', 'red') == red


def test_term_survives_copy_and_pickle():
    """Test that copying and pickling a term return a term equal to the registered one."""
    assert copy.deepcopy(LawKind.RATE) == LawKind.RATE
    assert copy.copy(SpaceKind.MESH) == SpaceKind.MESH
    assert pickle.loads(pickle.dumps(Form.SDE)) == Form.SDE  # noqa: S301  # nosec B301: loads only what it dumped


def test_term_value_cannot_be_set():
    """Test that the value of a term is read only."""
    with pytest.raises(AttributeError, match='value'):
        Trait.FORMLESS.value = 'other'


def test_term_text_and_name():
    """Test that a term prints its value and knows its registered name."""
    assert str(LawKind.LOCAL_RULE) == 'local-rule'
    assert Form.SDE.name == 'SDE'
    assert Form.MONTE_CARLO.value == 'Monte Carlo sampling'


def test_declared_noise_support():
    """Test that Euler-Maruyama and Heun declare every noise structure and Milstein only diagonal noise."""
    assert set(EulerMaruyamaSolver.supports.noise) == set(NoiseKind)
    assert set(HeunSolver.supports.noise) == set(NoiseKind)
    assert set(MilsteinSolver.supports.noise) == {NoiseKind.DIAGONAL}


RotorComponent = ElementComponent.define(
    'RotorComponent',
    states={'theta': state(0.0, domain=NumericDomain.ANGLE)},
    parameters={'omega': 1.0},
)


SpinInteraction = BaseInteraction.define(
    'SpinInteraction',
    roles=['target'],
    laws=lambda s: [RateLaw(s.target.theta, s.target.omega)],
)


SwitchComponent = ElementComponent.define('SwitchComponent', states={'on': state(False, domain=NumericDomain.BOOLEAN)})


@pytest.mark.parametrize(
    ('domain', 'value', 'expected'),
    [
        (NumericDomain.REAL, -1.5, True),
        (NumericDomain.POSITIVE, 0.0, False),
        (NumericDomain.NONNEGATIVE, 0.0, True),
        (NumericDomain.INTEGER, 2.5, False),
        (NumericDomain.NONNEGATIVE_INTEGER, [0, 3], True),
        (NumericDomain.REAL, float('nan'), False),
        (NumericDomain.REAL, 'm', False),
    ],
)
def test_domain_contains_checks_values(domain, value, expected):
    """Test that domains accept and reject values."""
    assert domain.contains(value) is expected


def test_domain_maps_to_sage_domain():
    """Test that domains give the Sage domains of their symbols, which have no nonnegative domains."""
    assert NumericDomain.NONNEGATIVE_INTEGER.sage_domain == 'integer'


def test_categorical_values_round_trip_as_labels():
    """Test that categorical states take labels as values and return them as labels."""
    system = System(
        components=[PersonComponent('ann', health='infected')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    trajectory = Simulation(system).run(until=1).trajectory_
    assert trajectory.values('ann.health').tolist() == ['infected', 'recovered']


def test_categorical_rejects_unknown_label():
    """Test that an initial value outside the alphabet is rejected."""
    with pytest.raises(ValidationError, match='categorical'):
        Simulation(System(components=[PersonComponent('ann', health='dead')])).run(until=1)


def test_angle_wraps_into_one_turn():
    """Test that an angle growing past π lies in (-π, π] after every frame."""
    system = System(
        components=[RotorComponent('r', theta=6.0, omega=1.0)],
        interactions=[SpinInteraction('s', target='r')],
    )
    values = Simulation(system, dt=0.5).run(until=3).trajectory_.values('r.theta')
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(9.0 - 2 * np.pi, abs=1e-6)


def test_angle_swings_around_zero_without_jumping():
    """Test that an angle decreasing through zero stays near zero instead of jumping a full turn."""
    system = System(
        components=[RotorComponent('r', theta=0.1, omega=-1.0)],
        interactions=[SpinInteraction('s', target='r')],
    )
    values = Simulation(system, dt=0.1).run(until=1).trajectory_.values('r.theta')
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(-0.9, abs=1e-6)


def test_angle_encode_wraps_into_half_open_turn():
    """Test that encoding an angle wraps it into (-π, π], keeping π and moving -π to π."""
    encoded = NumericDomain.ANGLE.encode([3 * np.pi / 2, -np.pi, np.pi])
    np.testing.assert_allclose(encoded, [-np.pi / 2, np.pi, np.pi])


def test_boolean_values_return_as_booleans():
    """Test that boolean states return booleans and reject other numbers."""
    trajectory = Simulation(System(components=[SwitchComponent('s', on=True)]), dt=1).run(until=1).trajectory_
    assert trajectory.values('s.on').tolist() == [True, True]
    with pytest.raises(ValidationError, match='boolean'):
        Simulation(System(components=[SwitchComponent('s', on=0.5)])).run(until=1)


class ValveComponent(ElementComponent):
    """Element with a bounded control input."""

    opening: float = parameter(0.5, adaptive=True, bounds=(0, 1))


def test_parameter_bounds_declare_a_bounded_adaptive_parameter():
    """Test that bounds of an adaptive parameter are stored as floats on its state declaration."""
    declaration = declarations_of(ValveComponent, VariableKind.STATE)['opening']
    assert declaration.adaptive
    assert declaration.bounds == (0.0, 1.0)
    assert isinstance(declaration.bounds[0], float)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'bounds': (0, 1)}, 'Only an adaptive parameter takes bounds'),
        ({'adaptive': True, 'bounds': (1, 0)}, 'lower at most upper'),
    ],
)
def test_parameter_bounds_reject_invalid_declarations(arguments, message):
    """Test that bounds without adaptivity or with a lower bound above the upper one are rejected."""
    with pytest.raises(ValueError, match=message):
        parameter(0.5, **arguments)


class BodyComponent(ElementComponent):
    """Element with one scalar state, one vector state, and one parameter."""

    n: float = state(1.0, domain=NumericDomain.NONNEGATIVE)
    x: list[float] = state([0.0, 0.0], vector=True)
    m: float = parameter(1.0, domain=NumericDomain.POSITIVE)


class LinkInteraction(BaseInteraction):
    """Interaction with two roles."""

    first: str = role()
    second: str = role()


def test_declarations_of_returns_fields_by_kind():
    """Test that declarations are grouped by kind in declaration order."""
    assert list(declarations_of(BodyComponent, VariableKind.STATE)) == ['n', 'x']
    assert list(declarations_of(BodyComponent, VariableKind.PARAMETER)) == ['m']
    assert list(declarations_of(LinkInteraction, VariableKind.ROLE)) == ['first', 'second']


def test_state_default_sequence_is_copied():
    """Test that a sequence default is not shared between instances."""
    first, second = BodyComponent('a'), BodyComponent('b')
    first.x.append(1.0)
    assert second.x == [0.0, 0.0]


def test_state_accepts_fractional_order_without_companions():
    """Test that a state of a fractional order keeps its order and gets no companion states."""
    RelaxingComponent = ElementComponent.define('RelaxingComponent', states={'x': state(1.0, order=0.6)})
    declaration = declarations_of(RelaxingComponent, VariableKind.STATE)['x']
    assert declaration.order == 0.6
    assert declaration.is_fractional
    assert declaration.derivatives == 0
    assert list(declarations_of(RelaxingComponent, VariableKind.STATE)) == ['x']


def test_state_normalizes_integer_valued_orders():
    """Test that an order with an integer value is an integer order with its companions."""
    declarations = declarations_of(
        ElementComponent.define('MassComponent', states={'x': state(0.0, order=2.0)}),
        VariableKind.STATE,
    )
    assert declarations['x'].order == 2
    assert not declarations['x'].is_fractional
    assert 'x_d1' in declarations


@pytest.mark.parametrize(
    ('arguments', 'error'),
    [
        ({'order': 0}, ValueError),
        ({'order': -0.5}, ValueError),
        ({'order': float('nan')}, ValueError),
        ({'order': 0.5, 'lags': 1}, ValueError),
        ({'order': True}, TypeError),
        ({'order': '0.5'}, TypeError),
    ],
)
def test_state_rejects_invalid_orders(arguments, error):
    """Test that an order that is not positive, not a number, or given with lags is rejected."""
    with pytest.raises(error, match='order'):
        state(1.0, **arguments)
