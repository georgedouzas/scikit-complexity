"""Test the lowering of encounter laws to gain and loss rates of densities on activity lattices, and validation."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement, NonnegativeElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BilinearLaw
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec

CLASSES = 9


def _exchange_table(size, share=0.5):
    """Return the transition table where a candidate moves one class toward the field particle with a probability."""
    table = np.zeros((size, size, size))
    for candidate in range(size):
        for field in range(size):
            step = int(np.sign(field - candidate))
            table[candidate, field, candidate + step] += share
            table[candidate, field, candidate] += 1 - share
    return table


class ExchangeInteraction(Interaction):
    """Conservative encounters of a population with itself, as a conserved bilinear law with an outcome table."""

    def __init__(
        self,
        label: str,
        *,
        population: Annotated[str, RoleSpec()],
        eta: Annotated[float, ParameterSpec()] = 1.0,
        table: np.ndarray | None = None,
    ) -> None:
        super().__init__(label)
        self.population = population
        self.eta = eta
        self.table = table

    def _get_laws(self, s):
        """Return the encounter."""
        return [BilinearLaw(s.population.x, s.population.x, s.population.x, s.eta, self.table, conserved=True)]


class RedistributionInteraction(Interaction):
    """Linear redistribution of a population from its first class to its second at rate one."""

    def __init__(self, label: str, *, population: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label)
        self.population = population

    def _get_laws(self, s):
        """Return the conserved bilinear law without a second density."""
        outcomes = np.asarray([[[0.0, 1.0]], [[0.0, 1.0]]])
        return [BilinearLaw(s.population.x, s.population.x, None, 1.0, outcomes, conserved=True)]


class GrowthInteraction(Interaction):
    """Proliferation of a population by encounters with a field population, and its destruction by them."""

    def __init__(
        self,
        label: str,
        *,
        population: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        mu: Annotated[float, ParameterSpec()] = 0.5,
        kill: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.population = population
        self.field = field
        self.mu = mu
        self.kill = kill

    def _get_laws(self, s):
        """Return the proliferation and the destruction."""
        return [
            BilinearLaw(s.population.x, s.population.x, s.field.x, s.mu),
            BilinearLaw(s.population.x, s.population.x, s.field.x, -s.kill),
        ]


def _profile(size=CLASSES):
    """Return a normalised initial profile peaked at a low activity."""
    grid = np.linspace(0.0, 1.0, size)
    profile = np.exp(-((grid - 0.3) ** 2) / 0.02)
    return profile / profile.sum()


def _wealth_system(table=None, time=None, space=None):
    """Return a system of one density on an activity lattice with conservative encounters."""
    return System(
        elements=[GroupElement('wealth', element=NonnegativeElement, on='sites', values={'x': _profile()})],
        interactions=[
            ExchangeInteraction(
                'exchange',
                population='wealth',
                table=table if table is not None else _exchange_table(CLASSES),
            ),
        ],
        space=space if space is not None else LatticeSpace(CLASSES, boundary='fixed'),
        time=time if time is not None else ContinuousTime(),
    )


def test_conservative_encounters_keep_the_density():
    """Test that conservative encounters keep the total density within 1e-10 over 100 time units."""
    trajectory = simulate(_wealth_system(), until=100, dt=1.0)
    totals = np.asarray(trajectory.values('wealth.x')).sum(axis=1)
    np.testing.assert_allclose(totals, totals[0], rtol=1e-10)
    assert not np.allclose(trajectory.values('wealth.x')[-1], _profile())


def test_encounter_rates_follow_gain_and_loss():
    """Test that the drift of an encounter is the gain minus the loss of Bellomo's equations."""
    problem = PassCompiler(backend=NumpyBackend()).prepare(_wealth_system()).run().problem_
    f = _profile()
    table = _exchange_table(CLASSES)
    expected = np.einsum('hki,h,k->i', table, f, f) - f * f.sum()
    np.testing.assert_allclose(problem.drift(0.0, problem.initial_state, problem.initial_parameters), expected)
    assert problem.jacobian is None
    assert problem.encounters[0].kind == 'transition'


def test_conserved_bilinear_law_without_a_second_density_redistributes_linearly():
    """Test that a linear redistribution at rate one empties the first class as exp(-t) and keeps the total."""
    system = System(
        elements=[GroupElement('mass', element=NonnegativeElement, on='sites', values={'x': [1.0, 0.0]})],
        interactions=[RedistributionInteraction('move', population='mass')],
        space=LatticeSpace(2, boundary='fixed'),
        time=ContinuousTime(),
    )
    trajectory = simulate(system, until=3, dt=0.5, solver=RK4Solver(dt=1e-3))
    values = np.asarray(trajectory.values('mass.x'))
    np.testing.assert_allclose(values[:, 0], np.exp(-np.asarray(trajectory.times)), atol=1e-6)
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-12)


def test_numba_backend_fuses_the_gain():
    """Test that the Numba backend runs the gain in a fused loop that agrees with NumPy."""
    numpy_problem = PassCompiler(backend=NumpyBackend()).prepare(_wealth_system()).run().problem_
    numba_problem = PassCompiler(backend=NumbaBackend()).prepare(_wealth_system()).run().problem_
    assert numba_problem.encounters[0].fused
    x, p = numpy_problem.initial_state, numpy_problem.initial_parameters
    np.testing.assert_allclose(numba_problem.drift(0.0, x, p), numpy_problem.drift(0.0, x, p), rtol=1e-12)


def test_encounter_rate_reads_its_parameter():
    """Test that a parameter in the encounter rate scales the rates at its current value."""
    problem = PassCompiler(backend=NumpyBackend()).prepare(_wealth_system()).run().problem_
    x, p = problem.initial_state, problem.initial_parameters.copy()
    base = problem.drift(0.0, x, p)
    p[problem.parameters.entry('exchange.eta').start] = 3.0
    np.testing.assert_allclose(problem.drift(0.0, x, p), 3 * base)


def test_parameter_in_transition_table_is_evaluated():
    """Test that a transition table written with a parameter takes the parameter's value."""

    class BiasedInteraction(Interaction):
        """Encounter whose transition probability is a parameter."""

        def __init__(
            self,
            label: str,
            *,
            population: Annotated[str, RoleSpec()],
            q: Annotated[float, ParameterSpec()] = 0.25,
        ) -> None:
            super().__init__(label)
            self.population = population
            self.q = q

        def _get_laws(self, s):
            """Return the encounter."""
            table = np.zeros((CLASSES, CLASSES, CLASSES), dtype=object)
            for candidate in range(CLASSES):
                table[candidate, :, min(candidate + 1, CLASSES - 1)] += s.q
                table[candidate, :, candidate] += 1 - s.q
            return [BilinearLaw(s.population.x, s.population.x, s.population.x, 1.0, table, conserved=True)]

    system = System(
        elements=[GroupElement('wealth', element=NonnegativeElement, on='sites', values={'x': _profile()})],
        interactions=[BiasedInteraction('bias', population='wealth')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_
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
        elements=[
            GroupElement('cells', element=NonnegativeElement, on='sites', values={'x': _profile()}),
            GroupElement('signal', element=NonnegativeElement, on='sites', values={'x': np.full(CLASSES, 0.1)}),
        ],
        interactions=[GrowthInteraction('growth', population='cells', field='signal', mu=0.5, kill=0.2)],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_
    x, p = problem.initial_state, problem.initial_parameters
    rates = problem.drift(0.0, x, p)
    cells = problem.states.entry('cells.x')
    signal_total = 0.1 * CLASSES
    np.testing.assert_allclose(rates[cells.indices], (0.5 - 0.2) * _profile() * signal_total)
    assert [item.kind for item in problem.encounters] == ['bilinear', 'bilinear']


def test_encounter_reads_a_field_on_a_grid_of_another_size():
    """Test that a candidate density reads a field density on an activity grid of another size."""
    system = System(
        elements=[
            GroupElement('cells', element=NonnegativeElement, on='sites', size=CLASSES, values={'x': _profile()}),
            GroupElement(
                'signal',
                element=NonnegativeElement,
                on='sites',
                size=3,
                space=LatticeSpace(3),
                values={'x': 0.1},
            ),
        ],
        interactions=[GrowthInteraction('growth', population='signal', field='cells')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_
    assert problem.encounters[0].test.size == 3


def test_invalid_transition_is_reported_before_compilation():
    """Test that transition probabilities summing to 0.9 are reported with the encounter, activities, and sum."""
    table = _exchange_table(CLASSES)
    table[2, 5] *= 0.9
    report = PassCompiler().prepare(_wealth_system(table=table)).report_
    issue = next(item for item in report.errors if item.code == 'encounter-probabilities')
    assert issue.path == 'exchange'
    assert 'candidate activity 2 with field activity 5 sums to 0.9' in issue.message
    with pytest.raises(ValidationError, match='do not sum to one'):
        PassCompiler().prepare(_wealth_system(table=table)).run()


def test_encounter_in_discrete_time_is_reported():
    """Test that an encounter in discrete time is reported with the fix of declaring continuous time."""
    report = PassCompiler().prepare(_wealth_system(time=DiscreteTime())).report_
    issue = next(item for item in report.errors if item.code == 'law-time')
    assert 'ContinuousTime' in issue.message
    assert '`ContinuousTime()`' in issue.fix


def test_rk4_runs_encounters():
    """Test that a fixed-step Runge-Kutta solver runs encounters like the default solver."""
    first = simulate(_wealth_system(), until=5, dt=1.0).values('wealth.x')[-1]
    second = simulate(_wealth_system(), until=5, dt=1.0, solver=RK4Solver(dt=0.01)).values('wealth.x')[-1]
    np.testing.assert_allclose(first, second, atol=1e-5)


def test_encounter_without_a_lattice_is_reported():
    """Test that an encounter of groups outside a lattice space is reported with the fix of declaring one."""
    system = System(
        elements=[GroupElement('wealth', element=NonnegativeElement, size=CLASSES, values={'x': _profile()})],
        interactions=[ExchangeInteraction('exchange', population='wealth', table=_exchange_table(CLASSES))],
        time=ContinuousTime(),
    )
    issue = next(item for item in PassCompiler().prepare(system).report_.errors if item.code == 'law-space')
    assert 'LatticeSpace' in issue.message
