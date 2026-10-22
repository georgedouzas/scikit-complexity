"""Test the lowering of encounter laws to gain and loss rates of densities on activity lattices, and validation."""

import numpy as np
import pytest

from skcomplex.core.base import Domain, ValidationError, parameter, role, state
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.components import Element, Group
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Destruction, Encounter, Proliferation
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.space import ContinuousTime, DiscreteTime, LatticeSpace
from skcomplex.core.system import System

CLASSES = 9


class Density(Element):
    """Density of particles at one activity value."""

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)


def _exchange_table(size, share=0.5):
    """Return the transition table where a candidate moves one class toward the field particle with a probability."""
    table = np.zeros((size, size, size))
    for candidate in range(size):
        for field in range(size):
            step = int(np.sign(field - candidate))
            table[candidate, field, candidate + step] += share
            table[candidate, field, candidate] += 1 - share
    return table


def _exchange(table):
    """Return an interaction of conservative encounters of a population with itself, with a transition table."""

    class Exchange(Interaction):
        """Conservative encounters of a population with itself."""

        population: str = role()
        eta: float = parameter(default=1.0)

        def laws(self, s):
            """Return the encounter."""
            return [Encounter(s.population.f, s.population.f, s.eta, table)]

    return Exchange


class Growth(Interaction):
    """Proliferation of a population by encounters with a field population, and its destruction by them."""

    population: str = role()
    field: str = role()
    mu: float = parameter(default=0.5)
    kill: float = parameter(default=0.0)

    def laws(self, s):
        """Return the proliferation and the destruction."""
        return [
            Proliferation(s.population.f, s.field.f, s.mu),
            Destruction(s.population.f, s.field.f, s.kill),
        ]


def _profile(size=CLASSES):
    """Return a normalised initial profile peaked at a low activity."""
    grid = np.linspace(0.0, 1.0, size)
    profile = np.exp(-((grid - 0.3) ** 2) / 0.02)
    return profile / profile.sum()


def _wealth_system(table=None, time=None, space=None):
    """Return a system of one density on an activity lattice with conservative encounters."""
    return System(
        components=[Group('wealth', element=Density, on='sites', values={'f': _profile()})],
        interactions=[
            _exchange(table if table is not None else _exchange_table(CLASSES))('exchange', population='wealth'),
        ],
        space=space if space is not None else LatticeSpace(CLASSES, boundary='fixed'),
        time=time if time is not None else ContinuousTime(),
    )


def test_conservative_encounters_keep_the_density():
    """Test that conservative encounters keep the total density within 1e-10 over 100 time units."""
    simulation = Simulation(_wealth_system(), dt=1.0).run(until=100)
    totals = np.asarray(simulation.trajectory_['wealth.f']).sum(axis=1)
    np.testing.assert_allclose(totals, totals[0], rtol=1e-10)
    assert not np.allclose(simulation.trajectory_['wealth.f'][-1], _profile())


def test_encounter_rates_follow_gain_and_loss():
    """Test that the drift of an encounter is the gain minus the loss of Bellomo's equations."""
    problem = compile_system(_wealth_system(), backend='numpy')
    f = _profile()
    table = _exchange_table(CLASSES)
    expected = np.einsum('hki,h,k->i', table, f, f) - f * f.sum()
    np.testing.assert_allclose(problem.drift(0.0, problem.initial_state, problem.initial_parameters), expected)
    assert problem.jacobian is None
    assert problem.encounters[0].kind == 'transition'


def test_numba_backend_fuses_the_gain():
    """Test that the Numba backend runs the gain in a fused loop that agrees with NumPy."""
    numpy_problem = compile_system(_wealth_system(), backend='numpy')
    numba_problem = compile_system(_wealth_system(), backend='numba')
    assert numba_problem.encounters[0].fused
    x, p = numpy_problem.initial_state, numpy_problem.initial_parameters
    np.testing.assert_allclose(numba_problem.drift(0.0, x, p), numpy_problem.drift(0.0, x, p), rtol=1e-12)


def test_encounter_rate_reads_its_parameter():
    """Test that a parameter in the encounter rate scales the rates at its current value."""
    problem = compile_system(_wealth_system(), backend='numpy')
    x, p = problem.initial_state, problem.initial_parameters.copy()
    base = problem.drift(0.0, x, p)
    p[problem.parameters.entry('exchange.eta').start] = 3.0
    np.testing.assert_allclose(problem.drift(0.0, x, p), 3 * base)


def test_parameter_in_transition_table_is_evaluated():
    """Test that a transition table written with a parameter takes the parameter's value."""

    class Biased(Interaction):
        """Encounter whose transition probability is a parameter."""

        population: str = role()
        q: float = parameter(default=0.25)

        def laws(self, s):
            """Return the encounter."""
            table = np.zeros((CLASSES, CLASSES, CLASSES), dtype=object)
            for candidate in range(CLASSES):
                table[candidate, :, min(candidate + 1, CLASSES - 1)] += s.q
                table[candidate, :, candidate] += 1 - s.q
            return [Encounter(s.population.f, s.population.f, 1.0, table)]

    system = System(
        components=[Group('wealth', element=Density, on='sites', values={'f': _profile()})],
        interactions=[Biased('bias', population='wealth')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = compile_system(system, backend='numpy')
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    f = _profile()
    expected = np.zeros(CLASSES)
    for candidate in range(CLASSES - 1):
        expected[candidate + 1] += 0.25 * f[candidate]
        expected[candidate] -= 0.25 * f[candidate]
    np.testing.assert_allclose(rates, expected, atol=1e-15)


def test_proliferation_and_destruction_change_the_density():
    """Test that proliferation adds and destruction removes density at the declared rates."""
    system = System(
        components=[
            Group('cells', element=Density, on='sites', values={'f': _profile()}),
            Group('signal', element=Density, on='sites', values={'f': np.full(CLASSES, 0.1)}),
        ],
        interactions=[Growth('growth', population='cells', field='signal', mu=0.5, kill=0.2)],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = compile_system(system, backend='numpy')
    x, p = problem.initial_state, problem.initial_parameters
    rates = problem.drift(0.0, x, p)
    cells = problem.states.entry('cells.f')
    signal_total = 0.1 * CLASSES
    np.testing.assert_allclose(rates[cells.indices], (0.5 - 0.2) * _profile() * signal_total)
    assert [item.kind for item in problem.encounters] == ['proliferation', 'destruction']


def test_encounter_reads_a_field_on_a_grid_of_another_size():
    """Test that a candidate density reads a field density on an activity grid of another size."""
    system = System(
        components=[
            Group('cells', element=Density, on='sites', size=CLASSES, values={'f': _profile()}),
            Group('signal', element=Density, on='sites', size=3, space=LatticeSpace(3), values={'f': 0.1}),
        ],
        interactions=[Growth('growth', population='signal', field='cells')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = compile_system(system, backend='numpy')
    assert problem.encounters[0].test.size == 3


def test_invalid_transition_is_reported_before_compilation():
    """Test that transition probabilities summing to 0.9 are reported with the encounter, activities, and sum."""
    table = _exchange_table(CLASSES)
    table[2, 5] *= 0.9
    report = validate(_wealth_system(table=table))
    issue = next(item for item in report.errors if item.code == 'encounter-probabilities')
    assert issue.path == 'exchange'
    assert 'candidate activity 2 with field activity 5 sums to 0.9' in issue.message
    with pytest.raises(ValidationError, match='do not sum to one'):
        compile_system(_wealth_system(table=table))


def test_encounter_in_discrete_time_is_reported():
    """Test that an encounter in discrete time is reported with the fix of declaring continuous time."""
    report = validate(_wealth_system(time=DiscreteTime()))
    issue = next(item for item in report.errors if item.code == 'law-time')
    assert 'continuous' in issue.message
    assert '`ContinuousTime()`' in issue.fix


def test_rk4_runs_encounters():
    """Test that a fixed-step Runge-Kutta solver runs encounters like the default solver."""
    first = Simulation(_wealth_system(), dt=1.0).run(until=5).trajectory_['wealth.f'][-1]
    second = Simulation(_wealth_system(), dt=1.0, solver=RK4(dt=0.01)).run(until=5).trajectory_['wealth.f'][-1]
    np.testing.assert_allclose(first, second, atol=1e-5)


def test_encounter_without_a_lattice_is_reported():
    """Test that an encounter of groups outside a lattice space is reported with the fix of declaring one."""
    system = System(
        components=[Group('wealth', element=Density, size=CLASSES, values={'f': _profile()})],
        interactions=[_exchange(_exchange_table(CLASSES))('exchange', population='wealth')],
        time=ContinuousTime(),
    )
    issue = next(item for item in validate(system).errors if item.code == 'law-space')
    assert 'lattice' in issue.message
