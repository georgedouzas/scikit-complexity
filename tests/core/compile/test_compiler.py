"""Test the compilation of systems into problems."""

import numpy as np
import pytest

from skcomplex.compile import compile_system
from skcomplex.core import (
    Form,
    IncompatibleLawsError,
    LabelError,
    RoleError,
    SpaceError,
    ValidationError,
    parameter,
    role,
    symbol,
)
from skcomplex.model import EuclideanSpace, Interaction, Rate, System, Update
from skcomplex.models.ecology import ExponentialGrowth, LogisticGrowth, Population, Predation
from skcomplex.models.economics import Cobweb, Market
from skcomplex.models.epidemiology import Compartment, Infection, Recovery
from skcomplex.models.finance import Asset, GeometricBrownianMotion
from skcomplex.models.physics import ConstantForce, HarmonicPotential, Particle


class CallableGrowth(Interaction):
    """Growth whose rate is a Python callable."""

    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        """Return a callable rate law."""
        name = f'{self.target}.n'
        return [Rate(s.target.n, lambda t, values: values[f'{self.label}.r'] * values[name])]


class Undeclared(Interaction):
    """Interaction that uses a symbol it does not declare."""

    target: str = role()

    def laws(self, s):
        """Return a law with an undeclared symbol."""
        return [Rate(s.target.n, symbol('k') * s.target.n)]


class DoubleUpdate(Interaction):
    """Interaction that updates the same state twice."""

    target: str = role()

    def laws(self, s):
        """Return two updates of one state."""
        return [Update(s.target.price, 1), Update(s.target.price, 2)]


def _ecology(growth):
    """Return a system of one population under a growth interaction."""
    return System(elements=[Population('rabbits', n=10)], interactions=[growth])


@pytest.mark.parametrize(
    ('system', 'form'),
    [
        (_ecology(LogisticGrowth('growth', target='rabbits')), Form.ODE),
        (
            System(
                elements=[Asset('stock', price=100)],
                interactions=[GeometricBrownianMotion('gbm', target='stock')],
            ),
            Form.SDE,
        ),
        (
            System(
                elements=[Compartment('I', n=5), Compartment('R')],
                interactions=[Recovery('rec', infected='I', recovered='R', stochastic=True)],
            ),
            Form.JUMP,
        ),
        (System(elements=[Market('m')], interactions=[Cobweb('c', target='m')]), Form.DISCRETE),
        (System(elements=[Population('a')]), Form.ODE),
    ],
)
def test_compile_system_infers_form(system, form):
    """Test that the form follows from the law kinds."""
    assert compile_system(system).form is form


def test_compile_system_mixed_kinds_raises_error():
    """Test that updates and rates in one system are rejected."""
    system = System(
        elements=[Market('m'), Population('a')],
        interactions=[Cobweb('c', target='m'), ExponentialGrowth('g', target='a')],
    )
    with pytest.raises(IncompatibleLawsError, match='Rate, Update'):
        compile_system(system)


def test_compile_system_names_nested_states_by_path():
    """Test that nested states are named by their path."""
    system = System(systems=[System('meadow', elements=[Population('rabbits', n=3)])])
    problem = compile_system(system)
    assert problem.states.names == ['meadow/rabbits.n']
    assert problem.initial_state.tolist() == [3.0]


def test_compile_system_couples_subsystems():
    """Test that an interaction couples elements of two subsystems."""
    system = System(
        systems=[
            System('forest', elements=[Population('foxes', n=2)]),
            System('meadow', elements=[Population('rabbits', n=5)]),
        ],
        interactions=[Predation('hunt', predator='forest/foxes', prey='meadow/rabbits', a=1.0, b=0.5)],
    )
    problem = compile_system(system)
    derivative = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert derivative.tolist() == [5.0, -10.0]


def test_compile_system_drift_takes_parameters():
    """Test that the drift reflects the parameter vector it receives."""
    problem = compile_system(_ecology(ExponentialGrowth('growth', target='rabbits', r=1.0)))
    x = problem.initial_state
    assert problem.drift(0.0, x, np.array([1.0])).tolist() == [10.0]
    assert problem.drift(0.0, x, np.array([2.0])).tolist() == [20.0]


def test_compile_system_callable_law_is_evaluated():
    """Test that a callable rate is added to the drift and left out of the equations."""
    problem = compile_system(_ecology(CallableGrowth('growth', target='rabbits', r=3.0)))
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [30.0]
    assert problem.has_callables
    assert problem.equations == {}


def test_compile_system_rates_add_up():
    """Test that two forces on one particle add up."""
    system = System(
        elements=[Particle('p', x=1.0, m=2.0)],
        interactions=[ConstantForce('push', target='p', F=4.0), HarmonicPotential('trap', target='p', k=6.0)],
        space=EuclideanSpace(1),
    )
    problem = compile_system(system)
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [0.0, -1.0]


def test_compile_system_keeps_oscillator_equations():
    """Test that the velocity equation of an oscillator is `-k * (x - center) / m`."""
    system = System(
        elements=[Particle('p')],
        interactions=[HarmonicPotential('trap', target='p')],
        space=EuclideanSpace(1),
    )
    problem = compile_system(system)
    x, v = problem.state_symbols
    k = problem.symbols['trap.k']
    m = problem.symbols['p.m']
    (center,) = problem.symbols['trap.center']
    assert problem.equations['p.x[0]'] == v
    assert (problem.equations['p.v[0]'] + k * (x - center) / m).simplify_full() == 0


def test_compile_system_vector_state_takes_space_dimension():
    """Test that vector states have one component per dimension."""
    problem = compile_system(System(elements=[Particle('p')], space=EuclideanSpace(2)))
    assert problem.states.component_names == ['p.x[0]', 'p.x[1]', 'p.v[0]', 'p.v[1]']


def test_compile_system_vector_state_without_space_raises_error():
    """Test that a vector state needs a space."""
    with pytest.raises(SpaceError, match='must have a space'):
        compile_system(System(elements=[Particle('p')]))


@pytest.mark.parametrize('m', [-2.0, 0.0, 'm'])
def test_compile_system_value_outside_domain_raises_error(m):
    """Test that a parameter outside its domain is rejected."""
    system = System(elements=[Particle('p', m=m)], space=EuclideanSpace(1))
    with pytest.raises(ValidationError, match=r'p\.m'):
        compile_system(system)


def test_compile_system_wrong_shape_raises_error():
    """Test that a vector value of the wrong size is rejected."""
    system = System(elements=[Particle('p', x=[1, 2, 3])], space=EuclideanSpace(2))
    with pytest.raises(ValidationError, match='shape'):
        compile_system(system)


def test_compile_system_unknown_role_raises_error():
    """Test that a role must name an element."""
    system = _ecology(LogisticGrowth('growth', target='foxes'))
    with pytest.raises(RoleError, match="'target' to 'foxes'"):
        compile_system(system)


def test_compile_system_duplicate_label_raises_error():
    """Test that siblings must have distinct labels."""
    system = System(elements=[Population('a'), Population('a')])
    with pytest.raises(LabelError, match="'a'"):
        compile_system(system)


def test_compile_system_invalid_label_raises_error():
    """Test that labels must be words."""
    with pytest.raises(LabelError, match='not valid'):
        compile_system(System(elements=[Population('a.b')]))


def test_compile_system_undeclared_symbol_raises_error():
    """Test that laws may only use declared symbols."""
    with pytest.raises(ValidationError, match='symbols k'):
        compile_system(_ecology(Undeclared('bad', target='rabbits')))


def test_compile_system_double_update_raises_error():
    """Test that a state takes one update law."""
    system = System(elements=[Market('m')], interactions=[DoubleUpdate('bad', target='m')])
    with pytest.raises(ValidationError, match='second time'):
        compile_system(system)


def test_compile_system_jump_stoichiometry():
    """Test that jump changes become rows of the stoichiometry matrix."""
    system = System(
        elements=[Compartment('S', n=9), Compartment('I', n=1)],
        interactions=[Infection('inf', susceptible='S', infected='I', beta=0.5, stochastic=True)],
    )
    problem = compile_system(system)
    assert problem.stoichiometry.tolist() == [[-1.0, 1.0]]
    assert problem.propensities(0.0, problem.initial_state, problem.initial_parameters).tolist() == [4.5]
