"""Test the validation of systems against the requirements of their laws and the support of solvers and backends."""

import importlib.util
import sys

import pytest

from skcomplex.core.backends import NumpyBackend
from skcomplex.core.base import Domain, OrderKind, Supports, ValidationError, VisualSpec, parameter, role, state
from skcomplex.core.compile import Issue, ValidationReport, ValidationWarning, validate
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Energy, Move, Noise, Objective, Rate, Rule, Sample, Sampler, Update, neighbors
from skcomplex.core.solvers import GrunwaldLetnikov
from skcomplex.core.space import DiscreteTime, GroupSpace, Memory, finitely_presented_group
from skcomplex.core.system import System


class Stock(Element):
    """Element with one nonnegative state."""

    n: float = state(1.0, domain=Domain.NONNEGATIVE)


class Opinion(Element):
    """Element with one categorical state."""

    view: float = state(0, domain=Domain.categorical('left', 'right'))


class Walker(Element):
    """Element with a vector position."""

    x: float = state(0.0, vector=True)


class Counter(Element):
    """Element whose state grows by one every step."""

    k: float = state(0.0)

    def laws(self, s):
        """Return the update of the count."""
        return [Update(s.k, s.k + 1)]


class Growth(Interaction):
    """Exponential growth of the state of its target."""

    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        """Return the rate of growth."""
        return [Rate(s.target.n, s.r * s.target.n)]


class Drift(Interaction):
    """Rate law on a categorical state, which no rate law may target."""

    target: str = role()

    def laws(self, s):
        """Return the rate of the opinion."""
        return [Rate(s.target.view, 1)]


class Ant(Element):
    """Agent that counts its steps."""

    steps: float = state(0.0)

    def laws(self, s):
        """Return the rule of the step."""
        return [Rule(self.walk)]

    def walk(self, agent, model):
        """Count one step."""
        agent.steps += 1


def _codes(report):
    """Return the codes of the issues of a report."""
    return [issue.code for issue in report.issues]


def test_valid_system_has_no_issues():
    """Test that a system of stocks with growth validates without issues."""
    system = System(components=[Stock('rabbits')], interactions=[Growth('growth', target='rabbits')])
    report = validate(system)
    assert report.ok
    assert report.issues == []


def test_several_errors_are_reported_at_once():
    """Test that an unbound role, a vector state without a space, and a rate on a categorical state are all reported."""
    system = System(
        components=[Stock('rabbits'), Walker('walker'), Opinion('voter')],
        interactions=[Growth('growth', target='foxes'), Drift('drift', target='voter')],
    )
    report = validate(system)
    assert sorted(_codes(report)) == ['law-domain', 'role', 'space-missing']
    assert {issue.path for issue in report.errors} == {'growth', 'walker', 'drift'}
    assert all(issue.fix for issue in report.issues)


def test_law_with_the_wrong_domain_suggests_laws():
    """Test that a rate law on a categorical state names the state and suggests jump or transition laws."""
    report = validate(System(components=[Opinion('voter')], interactions=[Drift('drift', target='voter')]))
    (issue,) = report.errors
    assert issue.code == 'law-domain'
    assert 'voter.view' in issue.message
    assert 'categorical' in issue.message
    assert 'jump' in issue.fix
    assert 'transition' in issue.fix


def test_unsupported_solver_lists_capable_solvers():
    """Test that a variable group validated with RK4 reports the unsupported components and capable solvers."""
    report = validate(System(components=[Ant.group('ants', 3, variable=True)]), solver='rk4')
    issue = next(issue for issue in report.errors if 'components' in issue.message)
    assert issue.code == 'solver-unsupported'
    assert 'variable-group' in issue.message
    assert 'mesa' in issue.fix


@pytest.mark.skipif(importlib.util.find_spec('lammps') is not None, reason='LAMMPS is installed')
def test_unavailable_solver_names_the_extra():
    """Test that choosing LAMMPS without it installed reports one error whose fix names the installation extra."""
    report = validate(System(components=[Stock('rabbits')]), solver='lammps')
    (issue,) = report.errors
    assert issue.code == 'solver-unavailable'
    assert '`lammps` extra' in issue.fix


def test_unknown_backend_is_reported():
    """Test that an unregistered backend is reported with the registered ones."""
    report = validate(System(components=[Stock('rabbits')]), backend='fortran')
    (issue,) = report.errors
    assert issue.code == 'backend-unknown'
    assert 'numpy' in issue.fix


def test_mixed_laws_suggest_partitions():
    """Test that rates and updates in one system without children are reported with both components."""
    system = System(components=[Stock('rabbits'), Counter('clock')], interactions=[Growth('growth', target='rabbits')])
    (issue,) = validate(system).errors
    assert issue.code == 'laws-mixed'
    assert 'rate: rabbits' in issue.message
    assert 'update: clock' in issue.message
    assert 'nested system' in issue.fix


def test_mixed_laws_in_nested_systems_are_valid():
    """Test that rates and updates in sibling nested systems validate as coupled partitions."""
    system = System(
        components=[
            System('meadow', components=[Stock('rabbits')], interactions=[Growth('growth', target='rabbits')]),
            System('clock', components=[Counter('counter')]),
        ],
    )
    assert validate(system).ok


def test_report_raises_every_error():
    """Test that raising a report lists every error with its path and code, and emits its warnings."""
    report = ValidationReport(
        [
            Issue('role', 'growth', 'The role is unbound.', 'Bind it.'),
            Issue('law-domain', 'drift', 'The law targets a categorical state.', 'Use a jump law.'),
            Issue('note', 'system', 'Something to know.', 'Nothing to do.', severity='warning'),
        ],
    )
    with pytest.warns(ValidationWarning, match='Something to know'), pytest.raises(ValidationError) as error:
        report.raise_errors()
    assert '2 error(s)' in str(error.value)
    assert 'growth: The role is unbound. Bind it. [role]' in str(error.value)
    assert '[law-domain]' in str(error.value)


class Relaxing(Element):
    """Element whose state relaxes with a Caputo derivative of order 0.6."""

    x: float = state(1.0, order=0.6)

    def laws(self, s):
        """Return the relaxation rate."""
        return [Rate(s.x, -s.x)]


class Overordered(Element):
    """Element whose state declares an order outside the fractional interval."""

    x: float = state(1.0, order=1.5)


class Remembering(Element):
    """Element with a state of a fractional order and no laws."""

    x: float = state(1.0, order=0.5)


class Shaken(Element):
    """Element whose fractional state is driven by noise."""

    x: float = state(1.0, order=0.6)

    def laws(self, s):
        """Return the noise on the state."""
        return [Noise(s.x, 0.1)]


def test_validate_fractional_state_with_rk4_names_fractional_solvers():
    """Test that a fractional state with the solver rk4 gives one error naming the state and the fractional solvers."""
    report = validate(System(components=[Relaxing('r')]), solver='rk4')
    assert [issue.code for issue in report.errors] == ['order-unsupported']
    assert 'r.x' in report.errors[0].message
    assert 'fractional-abm, grunwald-letnikov, l1' in report.errors[0].fix


def test_validate_order_out_of_range_names_interval():
    """Test that a state of order 1.5 gives one error naming the state and the interval of fractional orders."""
    report = validate(System(components=[Overordered('a')]))
    assert [issue.code for issue in report.errors] == ['fractional-order']
    assert 'a.x' in report.errors[0].message
    assert '(0, 1)' in report.errors[0].fix


def test_validate_fractional_state_under_discrete_time_suggests_continuous_time():
    """Test that a fractional state in a system of discrete time gives one error suggesting continuous time."""
    report = validate(System(components=[Remembering('a')], time=DiscreteTime()))
    assert [issue.code for issue in report.errors] == ['fractional-time']
    assert 'a.x' in report.errors[0].message
    assert 'continuous time' in report.errors[0].fix


def test_validate_fractional_state_with_backend_without_orders():
    """Test that a backend that declares no fractional orders is reported for a fractional state."""

    class IntegerBackend(NumpyBackend):
        """NumPy backend that integrates integer orders alone."""

        supports = Supports(orders={OrderKind.INTEGER})

    report = validate(System(components=[Relaxing('r')]), backend=IntegerBackend())
    assert [issue.code for issue in report.errors] == ['order-unsupported']
    assert 'numba, numpy' in report.errors[0].fix
    assert validate(System(components=[Relaxing('r')]), backend='numba').ok


def test_validate_noise_on_fractional_state_is_unsupported():
    """Test that a noise law on a fractional state is reported, since no fractional solver runs noise laws."""
    report = validate(System(components=[Shaken('a')]))
    assert [issue.code for issue in report.errors] == ['solver-unsupported']
    assert 'a.x' in report.errors[0].message


def test_validate_fractional_state_with_fractional_solver():
    """Test that a fractional state with a fractional solver and default settings is valid."""
    system = System(components=[Relaxing('r')])
    assert validate(system).ok
    assert validate(system, solver='l1').ok
    assert validate(system, solver=GrunwaldLetnikov()).ok


class Plant(Element):
    """Element steered by a bounded control input, with a quadratic cost."""

    x: float = state(1.0)
    u: float = parameter(0.5, adaptive=True, bounds=(0, 1))

    def laws(self, s):
        """Return the rate of the state and the cost."""
        return [Rate(s.x, s.u - s.x), Objective(s.x**2 + s.u**2)]


class NoisyPlant(Plant):
    """Plant whose state is driven by noise."""

    def laws(self, s):
        """Return the laws of the plant with noise."""
        return [*super().laws(s), Noise(s.x, 0.1)]


class Tank(Element):
    """Element whose objective is placed under discrete time."""

    level: float = state(1.0)

    def laws(self, s):
        """Return the update of the level and its cost."""
        return [Update(s.level, s.level / 2), Objective(s.level)]


def test_validate_control_outside_bounds():
    """Test that an adaptive parameter starting outside its bounds is one error at the parameter, naming its bounds."""
    (issue,) = validate(System(components=[Plant('plant', u=2.0)])).errors
    assert issue.path == 'plant.u'
    assert issue.code == 'parameter-bounds'
    assert '(0.0, 1.0)' in issue.fix


def test_validate_objective_under_discrete_time():
    """Test that an objective under discrete time is reported at its law, needing static or continuous time."""
    issues = validate(System(components=[Tank('tank')], time=DiscreteTime())).errors
    issue = next(issue for issue in issues if issue.code == 'law-time')
    assert issue.path == 'tank'
    assert 'Objective law' in issue.message
    assert 'continuous, static time' in issue.message


def test_validate_objective_with_a_solver_without_objectives():
    """Test that an objective run with a solver that does not optimise lists the solvers that do."""
    (issue,) = validate(System(components=[Plant('plant')]), solver='rk4').errors
    assert 'objective' in issue.message
    assert 'collocation, pontryagin' in issue.fix
    assert validate(System(components=[Plant('plant')]), solver='pontryagin').ok


def test_validate_noise_with_an_optimal_control_solver():
    """Test that a system with a noise law validated with Pontryagin reports that the solver lacks noise laws."""
    issues = validate(System(components=[NoisyPlant('plant')]), solver='pontryagin').errors
    issue = next(issue for issue in issues if issue.code == 'solver-unsupported')
    assert 'Pontryagin does not support the laws noise' in issue.message


class GroupSpin(Element):
    """Spin on a group space sampled with the default checkerboard schedule."""

    spin: float = state(1, domain=Domain.INTEGER)

    def laws(self, s):
        """Return the Glauber sampling of a ferromagnetic energy on the generators."""
        memory = Memory(('a', 'a^-1', 'b', 'b^-1'))
        energy = Energy(-s.spin * sum(neighbors(s.spin, memory)), memory)
        return [Sample(energy, Move.flip(s.spin), Sampler('glauber', 2.0))]


def test_group_space_reports_the_checkerboard_schedule():
    """Test that a sample with the checkerboard schedule on a group space is one issue naming the other schedules."""
    space = GroupSpace(finitely_presented_group(['a', 'b']), ball=2)
    issues = validate(System(components=[GroupSpin.lattice('spins', space)])).issues
    assert [(issue.code, issue.path) for issue in issues] == [('schedule-unsupported', 'spins')]
    assert "'synchronous'" in issues[0].fix
    assert "'random-sequential'" in issues[0].fix


def test_group_space_reports_an_image_visual():
    """Test that an image visual on a group space is reported with the Cayley visual as its fix."""
    space = GroupSpace(finitely_presented_group(['a', 'b']), ball=2)
    group = GroupSpin.lattice('spins', space, visual=VisualSpec('image'))
    issue = next(issue for issue in validate(System(components=[group])).issues if issue.code == 'visual-unsupported')
    assert "'cayley'" in issue.fix


def test_group_space_reports_a_ball_above_the_site_limit(monkeypatch):
    """Test that a ball of radius 12 of the free group of rank 3 is reported with the largest radius, without GAP."""
    monkeypatch.setitem(sys.modules, 'sage.all__sagemath_gap', None)
    space = GroupSpace(finitely_presented_group(['a', 'b', 'c']), ball=12)
    issues = validate(System(space=space)).issues
    assert [issue.code for issue in issues] == ['space-too-large']
    assert 'largest radius that fits is 6' in issues[0].message
