"""Test the validation of systems against the requirements of their laws and the support of solvers and backends."""

import importlib.util
import subprocess  # nosec B404: the test starts a new Python process without the fractional formalism
import sys

import pytest

import skcomplex.formalisms.control  # registers the optimal control solvers the tests name
import skcomplex.formalisms.particles  # noqa: F401  # registers the LAMMPS solver the tests name
from skcomplex.core.base import (
    BaseInteraction,
    Form,
    NoiseKind,
    NumericDomain,
    ValidationError,
    VisualSpec,
    parameter,
    role,
    state,
)
from skcomplex.core.components import ElementComponent, Wiener
from skcomplex.core.laws import NoiseLaw, Objective, RateLaw, UpdateLaw, neighbors
from skcomplex.core.space import DiscreteTime, LatticeSpace, Memory
from skcomplex.core.system import System
from skcomplex.engine.compile import SOLVERS, Issue, ValidationReport, ValidationWarning, compile_system, validate
from skcomplex.formalisms.agents import RuleLaw
from skcomplex.formalisms.groups import CayleySpace, finitely_presented_group
from skcomplex.formalisms.lattices import Energy, MoveLaw, SampleLaw, SamplerLaw

from ...conftest import HoldingComponent, _correlated
from .conftest import _pair


class StockComponent(ElementComponent):
    """Element with one nonnegative state."""

    n: float = state(1.0, domain=NumericDomain.NONNEGATIVE)


class OpinionComponent(ElementComponent):
    """Element with one categorical state."""

    view: float = state(0, domain=NumericDomain.categorical('left', 'right'))


class WalkerComponent(ElementComponent):
    """Element with a vector position."""

    x: float = state(0.0, vector=True)


class CounterComponent(ElementComponent):
    """Element whose state grows by one every step."""

    k: float = state(0.0)

    def laws(self, s):
        """Return the update of the count."""
        return [UpdateLaw(s.k, s.k + 1)]


class GrowthInteraction(BaseInteraction):
    """Exponential growth of the state of its target."""

    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        """Return the rate of growth."""
        return [RateLaw(s.target.n, s.r * s.target.n)]


class DriftInteraction(BaseInteraction):
    """Rate law on a categorical state, which no rate law may target."""

    target: str = role()

    def laws(self, s):
        """Return the rate of the opinion."""
        return [RateLaw(s.target.view, 1)]


class AntComponent(ElementComponent):
    """Agent that counts its steps."""

    steps: float = state(0.0)

    def laws(self, s):
        """Return the rule of the step."""
        return [RuleLaw(self.walk)]

    def walk(self, agent, model):
        """Count one step."""
        agent.set_state('steps', agent.state('steps') + 1)


def _codes(report):
    """Return the codes of the issues of a report."""
    return [issue.code for issue in report.issues]


def test_valid_system_has_no_issues():
    """Test that a system of stocks with growth validates without issues."""
    system = System(
        components=[StockComponent('rabbits')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    report = validate(system)
    assert report.ok
    assert report.issues == []


def test_several_errors_are_reported_at_once():
    """Test that an unbound role, a vector state without a space, and a rate on a categorical state are all reported."""
    system = System(
        components=[StockComponent('rabbits'), WalkerComponent('walker'), OpinionComponent('voter')],
        interactions=[GrowthInteraction('growth', target='foxes'), DriftInteraction('drift', target='voter')],
    )
    report = validate(system)
    assert sorted(_codes(report)) == ['law-domain', 'role', 'space-missing']
    assert {issue.path for issue in report.errors} == {'growth', 'walker', 'drift'}
    assert all(issue.fix for issue in report.issues)


def test_law_with_the_wrong_domain_suggests_laws():
    """Test that a rate law on a categorical state names the state and suggests jump or transition laws."""
    report = validate(
        System(components=[OpinionComponent('voter')], interactions=[DriftInteraction('drift', target='voter')]),
    )
    (issue,) = report.errors
    assert issue.code == 'law-domain'
    assert 'voter.view' in issue.message
    assert 'categorical' in issue.message
    assert 'jump' in issue.fix
    assert 'transition' in issue.fix


def test_unsupported_solver_lists_capable_solvers():
    """Test that a variable group validated with RK4 reports the unsupported components and capable solvers."""
    report = validate(System(components=[AntComponent.group('ants', 3, variable=True)]), solver='rk4')
    issue = next(issue for issue in report.errors if 'components' in issue.message)
    assert issue.code == 'solver-unsupported'
    assert 'variable-group' in issue.message
    assert 'mesa' in issue.fix


@pytest.mark.skipif(importlib.util.find_spec('lammps') is not None, reason='LAMMPS is installed')
def test_unavailable_solver_names_the_extra():
    """Test that choosing LAMMPS without it installed reports one error whose fix names the installation extra."""
    report = validate(System(components=[StockComponent('rabbits')]), solver='lammps')
    (issue,) = report.errors
    assert issue.code == 'solver-unavailable'
    assert '`lammps` extra' in issue.fix


def test_unknown_backend_is_reported():
    """Test that an unregistered backend is reported with the registered ones."""
    report = validate(System(components=[StockComponent('rabbits')]), backend='fortran')
    (issue,) = report.errors
    assert issue.code == 'backend-unknown'
    assert 'numpy' in issue.fix


def test_mixed_laws_suggest_partitions():
    """Test that rates and updates in one system without children are reported with both components."""
    system = System(
        components=[StockComponent('rabbits'), CounterComponent('clock')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    (issue,) = validate(system).errors
    assert issue.code == 'laws-mixed'
    assert 'rate: rabbits' in issue.message
    assert 'update: clock' in issue.message
    assert 'nested system' in issue.fix


def test_mixed_laws_in_nested_systems_are_valid():
    """Test that rates and updates in sibling nested systems validate as coupled partitions."""
    system = System(
        components=[
            System(
                'meadow',
                components=[StockComponent('rabbits')],
                interactions=[GrowthInteraction('growth', target='rabbits')],
            ),
            System('clock', components=[CounterComponent('counter')]),
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


class PlantComponent(ElementComponent):
    """Element steered by a bounded control input, with a quadratic cost."""

    x: float = state(1.0)
    u: float = parameter(0.5, adaptive=True, bounds=(0, 1))

    def laws(self, s):
        """Return the rate of the state and the cost."""
        return [RateLaw(s.x, s.u - s.x), Objective(s.x**2 + s.u**2)]


class NoisyPlantComponent(PlantComponent):
    """Plant whose state is driven by noise."""

    def laws(self, s):
        """Return the laws of the plant with noise."""
        return [*super().laws(s), NoiseLaw(s.x, 0.1)]


class TankComponent(ElementComponent):
    """Element whose objective is placed under discrete time."""

    level: float = state(1.0)

    def laws(self, s):
        """Return the update of the level and its cost."""
        return [UpdateLaw(s.level, s.level / 2), Objective(s.level)]


def test_validate_control_outside_bounds():
    """Test that an adaptive parameter starting outside its bounds is one error at the parameter, naming its bounds."""
    (issue,) = validate(System(components=[PlantComponent('plant', u=2.0)])).errors
    assert issue.path == 'plant.u'
    assert issue.code == 'parameter-bounds'
    assert '(0.0, 1.0)' in issue.fix


def test_validate_objective_under_discrete_time():
    """Test that an objective under discrete time is reported at its law, needing static or continuous time."""
    issues = validate(System(components=[TankComponent('tank')], time=DiscreteTime())).errors
    issue = next(issue for issue in issues if issue.code == 'law-time')
    assert issue.path == 'tank'
    assert 'Objective of tank' in issue.message
    assert 'continuous, static time' in issue.message


def test_validate_objective_with_a_solver_without_objectives():
    """Test that an objective run with a solver that does not optimise lists the solvers that do."""
    (issue,) = validate(System(components=[PlantComponent('plant')]), solver='rk4').errors
    assert 'objective' in issue.message
    assert 'collocation, pontryagin' in issue.fix
    assert validate(System(components=[PlantComponent('plant')]), solver='pontryagin').ok


def test_validate_noise_with_an_optimal_control_solver():
    """Test that a system with a noise law validated with Pontryagin reports that the solver lacks noise laws."""
    issues = validate(System(components=[NoisyPlantComponent('plant')]), solver='pontryagin').errors
    issue = next(issue for issue in issues if issue.code == 'solver-unsupported')
    assert 'PontryaginSolver does not support the laws noise' in issue.message


class GroupSpinComponent(ElementComponent):
    """Spin on a group space sampled with the default checkerboard schedule."""

    spin: float = state(1, domain=NumericDomain.INTEGER)

    def laws(self, s):
        """Return the Glauber sampling of a ferromagnetic energy on the generators."""
        memory = Memory(('a', 'a^-1', 'b', 'b^-1'))
        energy = Energy(-s.spin * sum(neighbors(s.spin, memory, reduce='each')), memory)
        return [SampleLaw(energy, MoveLaw.flip(s.spin), SamplerLaw('glauber', 2.0))]


def test_group_space_reports_the_checkerboard_schedule():
    """Test that a sample with the checkerboard schedule on a group space is one issue naming the other schedules."""
    space = CayleySpace(finitely_presented_group(['a', 'b']), ball=2)
    issues = validate(System(components=[GroupSpinComponent.lattice('spins', space)])).issues
    assert [(issue.code, issue.path) for issue in issues] == [('schedule-unsupported', 'spins')]
    assert "'synchronous'" in issues[0].fix
    assert "'random-sequential'" in issues[0].fix


def test_group_space_reports_an_image_visual():
    """Test that an image visual on a group space is reported with the Cayley visual as its fix."""
    space = CayleySpace(finitely_presented_group(['a', 'b']), ball=2)
    group = GroupSpinComponent.lattice('spins', space, visual=VisualSpec('image'))
    issue = next(issue for issue in validate(System(components=[group])).issues if issue.code == 'visual-unsupported')
    assert "'cayley'" in issue.fix


def test_group_space_reports_a_ball_above_the_site_limit(monkeypatch):
    """Test that a ball of radius 12 of the free group of rank 3 is reported with the largest radius, without GAP."""
    monkeypatch.setitem(sys.modules, 'sage.all__sagemath_gap', None)
    space = CayleySpace(finitely_presented_group(['a', 'b', 'c']), ball=12)
    issues = validate(System(space=space)).issues
    assert [issue.code for issue in issues] == ['space-too-large']
    assert 'largest radius that fits is 6' in issues[0].message


def test_lattice_with_a_bad_boundary_is_reported():
    """Test that a lattice whose boundary is not supported is a space-invalid error that lists the boundaries."""
    space = LatticeSpace((4, 4), boundary='spherical')
    (issue,) = validate(System(components=[StockComponent('rabbits')], space=space)).errors
    assert issue.code == 'space-invalid'
    assert "'spherical'" in issue.message
    assert 'periodic, fixed, reflecting' in issue.message


def test_invalid_space_among_other_errors_is_reported():
    """Test that an invalid lattice and an unbound role are both reported, and compiling raises the validation error."""
    system = System(
        components=[StockComponent('rabbits')],
        interactions=[GrowthInteraction('growth', target='hares')],
        space=LatticeSpace((4, 4), boundary='spherical'),
    )
    assert {issue.code for issue in validate(system).errors} == {'space-invalid', 'role'}
    with pytest.raises(ValidationError, match='space-invalid'):
        compile_system(system)


def test_validation_keeps_the_planned_compilation():
    """Test that validation returns the compilation state planned once, which holds the nodes and the plan."""
    system = System(
        components=[StockComponent('rabbits')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    compilation = validate(system).compilation
    assert [node.path for node in compilation.nodes] == ['rabbits', 'growth']
    assert [partition.name for partition in compilation.plan.partitions] == [system.label]


def test_validate_fractional_state_without_its_formalism_names_the_formalism():
    """Test that a fractional state with no fractional solver registered gives an error naming the formalism."""
    script = (
        'from skcomplex.core.base import state\n'
        'from skcomplex.engine.compile import validate\n'
        'from skcomplex.core.components import ElementComponent\n'
        'from skcomplex.core.system import System\n'
        "KindComponent = ElementComponent.define('KindComponent', states={'x': state(1.0, order=0.5)})\n"
        "issue = validate(System(components=[KindComponent('a')])).errors[0]\n"
        'print(issue.code, issue.fix)\n'
    )
    command = [sys.executable, '-c', script]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603  # nosec B603
    assert output.startswith('order-unsupported')
    assert 'skcomplex.formalisms.fractional' in output


class MisuseInteraction(BaseInteraction):
    """Rate law that uses the handle of a Wiener process as a value."""

    target: str = role()
    noise: str = role()

    def laws(self, s):
        """Return the rate."""
        return [RateLaw(s.target.n, s.noise.dW)]


class CoupledInteraction(BaseInteraction):
    """Stratonovich noise on a state, through a Wiener process another component also draws."""

    target: str = role()
    other: str = role()
    noise: str = role()

    def laws(self, s):
        """Return the noise of the target, whose amplitude reads the other state."""
        return [NoiseLaw(s.target.n, s.other.n, convention='stratonovich', process=s.noise.dW)]


class SingleInteraction(BaseInteraction):
    """Itô noise on one element from a Wiener process."""

    target: str = role()
    noise: str = role()

    def laws(self, s):
        """Return the noise."""
        return [NoiseLaw(s.target.n, 1.0, process=s.noise.dW)]


def test_validate_reports_mixed_conventions():
    """Test that laws of different conventions on one process are reported with both laws and the process."""
    issues = validate(System(components=[_pair(conventions=('ito', 'stratonovich'))('pair')])).issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-convention-mixed']
    assert 'ito NoiseLaw of pair' in issue.message
    assert 'stratonovich NoiseLaw of pair' in issue.message
    assert "'common'" in issue.message
    assert 'one convention' in issue.fix


def test_validate_reports_correction_out_of_scope():
    """Test that a Stratonovich law reading a state another component drives on its process is reported."""
    system = System(
        components=[HoldingComponent('a'), HoldingComponent('b', n=1.0), Wiener('market')],
        interactions=[
            CoupledInteraction('coupled', target='a', other='b', noise='market'),
            SingleInteraction('single', target='b', noise='market'),
        ],
    )
    (issue,) = [issue for issue in validate(system).issues if issue.code == 'noise-correction-scope']
    assert 'NoiseLaw of coupled' in issue.message
    assert 'NoiseLaw of single' in issue.message
    assert 'Itô' in issue.fix


def test_validate_reports_invalid_covariance():
    """Test that a covariance with a negative eigenvalue is reported with the element and the eigenvalue."""
    (issue,) = validate(_correlated([[1.0, 2.0], [2.0, 1.0]])).issues
    assert issue.code == 'wiener-covariance'
    assert issue.path == 'market'
    assert '-1' in issue.message


@pytest.mark.parametrize(
    'covariance',
    [[[1.0, 0.0, 0.0]], [[1.0, 0.5], [0.0, 1.0]]],
)
def test_validate_reports_covariance_shape_and_symmetry(covariance):
    """Test that a covariance that is not square of the dimension or not symmetric is reported."""
    assert [issue.code for issue in validate(_correlated(covariance)).issues] == ['wiener-covariance']


def test_validate_reports_handle_as_value():
    """Test that a Wiener handle used inside a rate law is reported with the law."""
    system = System(
        components=[HoldingComponent('a'), Wiener('market')],
        interactions=[MisuseInteraction('misuse', target='a', noise='market')],
    )
    (issue,) = [issue for issue in validate(system).issues if issue.code == 'wiener-handle']
    assert 'RateLaw of misuse' in issue.message
    assert 'only the process of a noise law' in issue.message


def test_validate_reports_wiener_under_discrete_time():
    """Test that a Wiener element in a system of discrete time is reported with the fix of continuous time."""
    CounterComponent = ElementComponent.define(
        'CounterComponent',
        states={'k': state(0.0, domain=NumericDomain.REAL)},
        laws=lambda s: [UpdateLaw(s.k, 1)],
    )
    system = System(components=[CounterComponent('counter'), Wiener('market')], time=DiscreteTime())
    (issue,) = [issue for issue in validate(system).issues if issue.code == 'wiener-time']
    assert issue.path == 'market'
    assert 'continuous time' in issue.fix


def test_validate_reports_milstein_with_shared_process():
    """Test that Milstein with a shared process gives one error that names the law and suggests other solvers."""
    (issue,) = validate(System(components=[_pair()('pair')]), solver='milstein').issues
    assert issue.code == 'noise-unsupported'
    assert 'NoiseLaw of pair' in issue.message
    assert 'euler-maruyama' in issue.fix
    assert 'heun' in issue.fix


def test_validate_reports_lammps_with_shared_process():
    """Test that the LAMMPS backend with a shared process is reported with the backends that support it."""
    issues = validate(System(components=[_pair()('pair')]), solver='lammps').issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-unsupported']
    assert 'NoiseLaw of pair' in issue.message
    assert 'numba, numpy' in issue.fix


def test_validate_accepts_correlated_noise_with_automatic_solver():
    """Test that the automatic SDE solver declares correlated noise, so validation accepts it."""
    assert validate(_correlated([[1.0, 0.5], [0.5, 1.0]])).ok
    assert NoiseKind.CORRELATED in SOLVERS.resolve((Form.SDE, 'auto')).supports.noise
