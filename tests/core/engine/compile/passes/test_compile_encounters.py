"""Test the lowering of encounter laws to gain and loss rates of densities on activity lattices, and validation."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import bilinear
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile.passes._compile_encounters import CompileEncountersPass
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

CLASSES = 9


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        size: int | None = None,
        space: LatticeSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x


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
        population: Role,
        eta: float = 1.0,
        table: np.ndarray | None = None,
    ) -> None:
        super().__init__(label)
        self.population = population
        self.eta = eta
        self.table = table

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the encounter."""
        super()._prepare(context)
        s = context
        self.laws_ = [
            RateLaw(s.population.x, bilinear(s.eta, self.table, s.population.x, s.population.x, conserved=True)),
        ]


class RedistributionInteraction(Interaction):
    """Linear redistribution of a population from its first class to its second at rate one."""

    def __init__(self, label: str, *, population: Role) -> None:
        super().__init__(label)
        self.population = population

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the conserved bilinear law without a second density."""
        super()._prepare(context)
        s = context
        outcomes = np.asarray([[[0.0, 1.0]], [[0.0, 1.0]]])
        self.laws_ = [RateLaw(s.population.x, bilinear(1.0, outcomes, s.population.x, conserved=True))]


class GrowthInteraction(Interaction):
    """Proliferation of a population by encounters with a field population, and its destruction by them."""

    def __init__(
        self,
        label: str,
        *,
        population: Role,
        field: Role,
        mu: float = 0.5,
        kill: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.population = population
        self.field = field
        self.mu = mu
        self.kill = kill

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the proliferation and the destruction."""
        super()._prepare(context)
        s = context
        self.laws_ = [
            RateLaw(s.population.x, bilinear(s.mu, None, s.population.x, s.field.x)),
            RateLaw(s.population.x, bilinear(-s.kill, None, s.population.x, s.field.x)),
        ]


def _profile(size=CLASSES):
    """Return a normalised initial profile peaked at a low activity."""
    grid = np.linspace(0.0, 1.0, size)
    profile = np.exp(-((grid - 0.3) ** 2) / 0.02)
    return profile / profile.sum()


def _wealth_system(table=None, space=None):
    """Return a system of one density on an activity lattice with conservative encounters."""
    space = space if space is not None else LatticeSpace(CLASSES, boundary='fixed')
    return System(
        elements=[AmountElement('wealth', x=_profile(), space=space)],
        interactions=[
            ExchangeInteraction(
                'exchange',
                population='wealth',
                table=table if table is not None else _exchange_table(CLASSES),
            ),
        ],
        space=space,
    )


def test_conservative_encounters_keep_the_density():
    """Test that conservative encounters keep the total density within 1e-10 over 100 time units."""
    trajectory = _wealth_system().prepare().simulate(until=100, dt=1.0)
    totals = np.asarray(trajectory.get_values('wealth.x')).sum(axis=1)
    np.testing.assert_allclose(totals, totals[0], rtol=1e-10)
    assert not np.allclose(trajectory.get_values('wealth.x')[-1], _profile())


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
    space = LatticeSpace(2, boundary='fixed')
    system = System(
        elements=[AmountElement('mass', x=[1.0, 0.0], space=space)],
        interactions=[RedistributionInteraction('move', population='mass')],
        space=space,
        solver=RK4Solver(dt=1e-3),
    )
    trajectory = system.prepare().simulate(until=3, dt=0.5)
    values = np.asarray(trajectory.get_values('mass.x'))
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
    p[problem.parameters.get_entry('exchange.eta').start] = 3.0
    np.testing.assert_allclose(problem.drift(0.0, x, p), 3 * base)


def test_parameter_in_transition_table_is_evaluated():
    """Test that a transition table written with a parameter takes the parameter's value."""

    class BiasedInteraction(Interaction):
        """Encounter whose transition probability is a parameter."""

        def __init__(
            self,
            label: str,
            *,
            population: Role,
            q: float = 0.25,
        ) -> None:
            super().__init__(label)
            self.population = population
            self.q = q

        def _prepare(self, context: SimpleNamespace) -> None:
            """Set the encounter."""
            super()._prepare(context)
            s = context
            table = np.zeros((CLASSES, CLASSES, CLASSES), dtype=object)
            for candidate in range(CLASSES):
                table[candidate, :, min(candidate + 1, CLASSES - 1)] += s.q
                table[candidate, :, candidate] += 1 - s.q
            self.laws_ = [
                RateLaw(s.population.x, bilinear(1.0, table, s.population.x, s.population.x, conserved=True)),
            ]

    space = LatticeSpace(CLASSES, boundary='fixed')
    system = System(
        elements=[AmountElement('wealth', x=_profile(), space=space)],
        interactions=[BiasedInteraction('bias', population='wealth')],
        space=space,
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
    space = LatticeSpace(CLASSES, boundary='fixed')
    system = System(
        elements=[
            AmountElement('cells', x=_profile(), space=space),
            AmountElement('signal', x=np.full(CLASSES, 0.1), space=space),
        ],
        interactions=[GrowthInteraction('growth', population='cells', field='signal', mu=0.5, kill=0.2)],
        space=space,
    )
    problem = PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_
    x, p = problem.initial_state, problem.initial_parameters
    rates = problem.drift(0.0, x, p)
    cells = problem.states.get_entry('cells.x')
    signal_total = 0.1 * CLASSES
    np.testing.assert_allclose(rates[cells.indices], (0.5 - 0.2) * _profile() * signal_total)
    assert [item.kind for item in problem.encounters] == ['bilinear', 'bilinear']


def test_encounter_reads_a_field_on_a_grid_of_another_size():
    """Test that a candidate density reads a field density on an activity grid of another size."""
    space = LatticeSpace(CLASSES, boundary='fixed')
    system = System(
        elements=[
            AmountElement('cells', x=_profile(), space=space),
            AmountElement('signal', x=np.full(3, 0.1), space=LatticeSpace(3)),
        ],
        interactions=[GrowthInteraction('growth', population='signal', field='cells')],
        space=space,
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


def test_rk4_runs_encounters():
    """Test that a fixed-step Runge-Kutta solver runs encounters like the default solver."""
    first = _wealth_system().prepare().simulate(until=5, dt=1.0).get_values('wealth.x')[-1]
    second = _wealth_system().set_params(solver=RK4Solver(dt=0.01)).prepare().simulate(until=5, dt=1.0)
    second = second.get_values('wealth.x')[-1]
    np.testing.assert_allclose(first, second, atol=1e-5)


def test_encounter_without_a_lattice_is_reported():
    """Test that an encounter of groups outside a lattice space is reported with the fix of declaring one."""
    system = System(
        elements=[AmountElement('wealth', x=_profile(), size=CLASSES)],
        interactions=[ExchangeInteraction('exchange', population='wealth', table=_exchange_table(CLASSES))],
    )
    issue = next(item for item in PassCompiler().prepare(system).report_.errors if item.code == 'law-space')
    assert 'LatticeSpace' in issue.message


def test_compile_encounters_pass_get_kinetic_table():
    """Test `CompileEncountersPass._get_kinetic_table` on a small case."""
    assert CompileEncountersPass._get_kinetic_table([[0.5, 0.5]]).dtype == np.float64
