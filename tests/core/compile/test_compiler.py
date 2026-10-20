"""Test the compilation of systems into problems."""

import time

import numpy as np
import pytest

from skcomplex.core.base import (
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
from skcomplex.core.compile import compile_system
from skcomplex.core.model import (
    ConstantForce,
    Element,
    EuclideanSpace,
    HarmonicPotential,
    Interaction,
    LatticeSpace,
    Particle,
    Rate,
    System,
    Update,
)
from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.ecology import ExponentialGrowth, LogisticGrowth, Population, Predation
from skcomplex.models.biology.epidemiology import Compartment, Infection, Recovery
from skcomplex.models.economics.finance import Asset, GeometricBrownianMotion
from skcomplex.models.economics.markets import Cobweb, Market
from skcomplex.models.physics.statistical_mechanics import ParticleSite, ZeroRangeHop


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


Node = Element.define('Node', states={'x': 0.0})


class WeightedSum(Interaction):
    """Update of a target to the weighted sum of the states of a variable number of inputs."""

    target: str = role()
    inputs: list[str] = role(variadic=True)
    weights: list[float] = parameter(1.0, per='inputs')

    def laws(self, s):
        """Return the update of the target."""
        return [Update(s.target.x, sum((w * item.x for w, item in zip(s.weights, s.inputs, strict=True)), 0))]


def _summed(inputs, weights=1.0, **settings):
    """Return the system of a weighted sum of inputs read from three elements and a group of ten nodes."""
    elements = [
        Node('a', x=1.0),
        Node('b', x=2.0),
        Node('c', x=4.0),
        Node('hub', x=8.0),
        Node('target'),
        Node.group('nodes', 10, x=np.arange(10.0)),
    ]
    return System(elements=elements, interactions=[WeightedSum('sum', target='target', inputs=inputs, weights=weights)])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_compile_variadic_role_sums_any_number_of_inputs(backend):
    """Test that a variadic role binds a sequence of elements that laws read as a tuple of namespaces."""
    problem = compile_system(_summed(['a', 'b', 'c']), backend=backend)
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(state, 'target.x') == 7.0
    assert problem.interactions[0].roles['inputs'] == ('a', 'b', 'c')


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_compile_variadic_member_reference_reads_one_member(backend):
    """Test that a member reference of a variadic role reads the states of that member of a group."""
    problem = compile_system(_summed(['hub', 'nodes[3]'], weights=[0.0, 1.0]), backend=backend)
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(state, 'target.x') == 3.0


@pytest.mark.parametrize(
    ('inputs', 'match'),
    [
        (['a', 'nodes[12]'], r"sum .*'nodes\[12\]'.*'inputs'.*10 members"),
        (['a', 'nodes'], 'binds the group nodes'),
        (['a', 'missing'], "'missing'"),
        ('a', 'Bind a list'),
    ],
)
def test_compile_variadic_role_with_wrong_entry_raises_error(inputs, match):
    """Test that a variadic entry naming a missing member, a group, or no element raises a role error."""
    with pytest.raises(RoleError, match=match):
        compile_system(_summed(inputs))


Hub = Element.define('Hub', states={'y': 0.0})


class SiteDrive(Interaction):
    """Growth of a hub at the rate of the state of one site."""

    hub: str = role()
    site: str = role()

    def laws(self, s):
        """Return the growth of the hub."""
        return [Rate(s.hub.y, s.site.x)]


class SitesDrive(Interaction):
    """Growth of a hub at the rate of the summed states of several sites."""

    hub: str = role()
    sites: list[str] = role(variadic=True)

    def laws(self, s):
        """Return the growth of the hub."""
        return [Rate(s.hub.y, sum((item.x for item in s.sites), 0))]


def _hub_and_sites(interaction):
    """Return a hub and a group of five sites with states 0 to 4, coupled by an interaction."""
    return System(elements=[Node.group('sites', 5, x=np.arange(5.0)), Hub('hub')], interactions=[interaction])


def test_compile_plain_role_binds_one_group_member():
    """Test that a plain role bound to a member reference reads that member like a variadic entry does."""
    plain = compile_system(_hub_and_sites(SiteDrive('link', hub='hub', site='sites[3]')))
    variadic = compile_system(_hub_and_sites(SitesDrive('link', hub='hub', sites=['sites[3]'])))
    for problem in (plain, variadic):
        drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
        assert problem.states.read(drift, 'hub.y') == 3.0
    assert plain.interactions[0].roles['site'] == 'sites[3]'


def test_compile_plain_role_past_group_end_raises_error():
    """Test that a plain role bound past the end of a group names the interaction, role, reference, and size."""
    with pytest.raises(RoleError, match=r"link binds 'sites\[7\]' to its role 'site', but sites holds 5 members"):
        compile_system(_hub_and_sites(SiteDrive('link', hub='hub', site='sites[7]')))


def test_compile_per_entry_parameter_with_wrong_length_raises_error():
    """Test that a per-entry parameter of another length than its role names the interaction and both lengths."""
    with pytest.raises(ValidationError, match=r"sum .*'weights' 2 values.*'inputs' binds 3 entries"):
        compile_system(_summed(['a', 'b', 'c'], weights=[1.0, 2.0]))


def test_compile_per_entry_parameter_has_one_component_per_entry():
    """Test that a per-entry parameter is laid out with one addressable component per entry."""
    problem = compile_system(_summed(['a', 'b', 'c'], weights=[1.0, 2.0, 3.0]))
    assert problem.parameters.component_names == ['sum.weights[0]', 'sum.weights[1]', 'sum.weights[2]']


def test_simulation_updates_one_per_entry_weight_live():
    """Test that a live update of one component of a per-entry parameter changes that entry only."""
    schedule = {'sum.weights[1]': lambda t: 1.0 if t < 1 else 10.0}
    trajectory = Simulation(_summed(['a', 'b', 'c']), dt=1, schedule=schedule).run(until=2).trajectory_
    assert trajectory['target.x'].tolist() == [0.0, 7.0, 25.0]
    simulation = Simulation(_summed(['a', 'b', 'c'])).run(until=1)
    simulation.update_parameters({'sum.weights[2]': 5.0})
    assert simulation.parameters_.tolist() == [1.0, 1.0, 5.0]


def _compile_time(n_links):
    """Return the shortest of two compile times of a zero-range ring written with one interaction per link."""
    sites = ParticleSite.lattice('sites', LatticeSpace(n_links), n=np.ones(n_links))
    links = [
        ZeroRangeHop(f'hop_{index}', link=[f'sites[{index}]', f'sites[{(index + 1) % n_links}]'])
        for index in range(n_links)
    ]
    system = System(elements=[sites], interactions=links)
    durations = []
    for _ in range(2):
        start = time.perf_counter()
        compile_system(system, backend='numpy')
        durations.append(time.perf_counter() - start)
    return min(durations)


def test_compile_system_time_grows_linearly_with_interactions():
    """Test that compiling 800 one-link interactions takes at most 2.5 times as long as compiling 400."""
    assert _compile_time(800) <= 2.5 * _compile_time(400)
