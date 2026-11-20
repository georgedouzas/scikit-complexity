"""Test the general compiler: its settings, its report on a system, and the problem it runs to."""

import re
import time
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_gap import libgap
from sage.all__sagemath_symbolics import RR, SR, Expression, cos, identity_matrix, oo, sin, sqrt
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element, Tensor
from skcomplex.core.components.interactions import Interaction, WienerProcessSpec
from skcomplex.core.components.laws import (
    JumpLaw,
    NoiseLaw,
    ObjectiveLaw,
    RateLaw,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import (
    codifferential,
    convolve,
    coordinate,
    cumulative_integral,
    displacement,
    exterior_derivative,
    hodge_star,
    integral_transform,
    integration_variable,
    inverse_helmholtz,
    laplacian,
    mean,
    mod,
    neighbors,
    tensor_divergence,
    variance,
)
from skcomplex.core.components.spaces import (
    AxisSpec,
    BaseSpace,
    CellComplexSpace,
    CofaceNeighborhood,
    ContinuousSpace,
    LatticeSpace,
    NetworkSpace,
    PhaseSpace,
    RadiusNeighborhood,
)
from skcomplex.core.engine._utils import domain_of, symbol
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import BasePass, IssueSpec, ValidationReportSpec
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile.passes._flatten import FlattenPass
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.engine.problems import ProblemSpec
from skcomplex.core.engine.solvers import (
    GillespieSolver,
    LammpsSolver,
    MetropolisSolver,
    MilsteinSolver,
    PontryaginSolver,
    RK4Solver,
)
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.core.enums import FormKind, NoiseKind
from skcomplex.core.errors import (
    IncompatibleLawsError,
    LabelError,
    PipelineError,
    RoleElementError,
    RoleError,
    SolverError,
    SpaceError,
    UnregisteredLawError,
    ValidationError,
)
from skcomplex.core.specs import VisualSpec
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.common.populations import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import LotkaVolterraSystem, PredationInteraction
from skcomplex.systems.biology.epidemiology import InfectionInteraction, RecoveryInteraction
from skcomplex.systems.computer_science.common.automata import CayleySpace, CayleyVisual, MemoryNeighborhood
from skcomplex.systems.economics.finance import AssetElement, GeometricBrownianMotionInteraction
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
from skcomplex.systems.general.rewriting import ProductionLaw, RewritingInteraction, WordElement
from skcomplex.systems.physics.common.particles import ConstantForceInteraction
from skcomplex.systems.physics.common.quantum import QubitElement
from skcomplex.systems.physics.statistical_mechanics import ZeroRangeHopInteraction
from skcomplex.systems.physics.waves import HarmonicPotentialInteraction
from skcomplex.systems.social.cooperation import SimplexSpace

from ....conftest import (
    BodyElement,
    CallableGrowthInteraction,
    HoldingElement,
    NodeElement,
    _correlated,
    _summed,
    _weighted,
)
from .conftest import pair_system

CUTOFF = 2.5
RECEIVED: dict[str, object] = {}


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount, or one per member of a group."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x


def _rule(label: str, target: Role, make) -> Interaction:
    """Return an interaction with one role `target` whose laws `make` returns from the namespace."""
    return Interaction(label, roles={'target': target}, laws=make)


def _growth():
    """Return a system of one growing population."""
    return System(
        elements=[PopulationElement('rabbits', x=10)],
        interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.5)],
    )


def _pass(name, reads, writes, log=None):
    """Return a pass that records its name when it runs."""

    def _get_reads(self):
        """Return the fields the pass reads."""
        return reads

    def _get_writes(self):
        """Return the fields the pass writes."""
        return writes

    def _run(self, inputs=None):
        """Record the run."""
        if log is not None:
            log.append(name)

    namespace = {'_get_reads': _get_reads, '_get_writes': _get_writes, '_run': _run, '__doc__': name}
    return type(name, (BasePass,), namespace)()


def _names(passes):
    """Return the class names of passes."""
    return [type(item).__name__ for item in passes]


class StockElement(Element):
    """Element with one nonnegative value."""

    def __init__(
        self,
        label: str,
        *,
        n: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.n = n


class CounterElement(Element):
    """Element whose count `k` the interaction `_tick` raises by one every step."""

    def __init__(self, label: str, *, k: float = 0.0) -> None:
        super().__init__(label)
        self.k = k


def _tick(label: str, target: Role) -> Interaction:
    """Return the update that raises the count of its target by one every step."""
    return _rule(label, target, lambda s: [UpdateLaw(s.target.k, s.target.k + 1)])


class GrowthInteraction(Interaction):
    """Exponential growth of the value `n` of its target."""

    def __init__(self, label: str, *, target: Role, r: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of growth."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.target.n, context.r * context.target.n)]


def _codes(report):
    """Return the codes of the issues of a report."""
    return [issue.code for issue in report.issues]


class ScaledRateLaw(RateLaw):
    """A rate law of a class the engine has no form for."""


class PlantElement(Element):
    """Element steered by a control input `u` bounded to `[0, 1]`."""

    def __init__(
        self,
        label: str,
        *,
        x: float = 1.0,
        u: Annotated[float, RealSet.closed(0, 1)] = 0.5,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.u = u


def _plant(*, noisy: bool = False) -> Interaction:
    """Return the rate `u - x` of the plant whose control `u` minimises `x^2`, with noise if `noisy`."""

    def laws(s):
        """Return the rate, the cost, and the noise of the plant."""
        plant = s.target
        noise = [NoiseLaw(plant.x, 0.1)] if noisy else []
        return [RateLaw(plant.x, plant.u - plant.x), ObjectiveLaw(plant.x**2, controls=[plant.u]), *noise]

    return _rule('steer', 'plant', laws)


class MisuseInteraction(Interaction):
    """Rate law that uses the handle of a Wiener process as a value."""

    def __init__(self, label: str, *, target: Role, noise: WienerProcessSpec) -> None:
        super().__init__(label)
        self.target = target
        self.noise = noise

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.target.n, context.noise.dW)]


class CoupledInteraction(Interaction):
    """Stratonovich noise on a value, through a Wiener process another component also draws."""

    def __init__(self, label: str, *, target: Role, other: Role, noise: WienerProcessSpec) -> None:
        super().__init__(label)
        self.target = target
        self.other = other
        self.noise = noise

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise of the target, whose amplitude reads the other value."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [NoiseLaw(s.target.n, s.other.n, convention='stratonovich', process=s.noise.dW)]


class SingleInteraction(Interaction):
    """Itô noise on one element from a Wiener process."""

    def __init__(self, label: str, *, target: Role, noise: WienerProcessSpec) -> None:
        super().__init__(label)
        self.target = target
        self.noise = noise

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [NoiseLaw(context.target.n, 1.0, process=context.noise.dW)]


def _spread(layer: str | bool = True) -> Interaction:
    """Return the spread of the value of a node to its neighbours along the edges of a layer."""
    return Interaction(
        'spread',
        among='nodes',
        along=layer,
        laws=lambda s: [RateLaw(s.target.x, s.source.x - s.target.x)],
    )


def _consensus() -> Interaction:
    """Return the pull of the members of every hyperedge towards their mean."""
    return Interaction(
        'consensus',
        among='nodes',
        along='hyperedges',
        laws=lambda s: [RateLaw(s.member.x, mean(s.members.x, over=s.members) - s.member.x)],
    )


def _bindings(interaction, space):
    """Return the binding issues of a group of four nodes on a space, coupled by an interaction."""
    nodes = NodeElement('nodes', x=np.zeros(4), size=4, space=space)
    system = System(elements=[nodes], interactions=[interaction], space=space)
    report = PassCompiler().prepare(system).report_
    issues = report.issues
    return [issue for issue in issues if issue.code == 'binding']


def _reading(label: str = 'read') -> Interaction:
    """Return a rate that reads the value `n` of its target."""
    return _rule(label, 'tank', lambda s: [RateLaw(s.target.n, -s.target.n)])


class TankElement(Element):
    """Element whose only value is `level`."""

    def __init__(self, label: str, level: float = 1.0) -> None:
        super().__init__(label)
        self.level = level


def _role_issues(system: System) -> list[IssueSpec]:
    """Return the `role-element` issues of a system."""
    return [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'role-element']


def test_compiler_prepares_its_report_and_runs_to_the_problem():
    """Test that `prepare` sets the report of a system and `run` sets its problem."""
    compiler = PassCompiler().prepare(_growth())
    assert isinstance(compiler.report_, ValidationReportSpec)
    assert compiler.report_.ok
    problem = compiler.run({'rabbits.x': 20}).problem_
    assert isinstance(problem, ProblemSpec)
    assert problem.initial_state.tolist() == [20.0]


def test_compiler_run_raises_the_errors_of_its_report():
    """Test that `run` raises the errors that `prepare` found, after `prepare` reported them."""
    compiler = PassCompiler(solver=GillespieSolver()).prepare(_growth())
    assert [issue.code for issue in compiler.report_.errors] == ['solver-unsupported']
    with pytest.raises(SolverError, match='solver-unsupported'):
        compiler.run()


@pytest.mark.parametrize(
    'settings',
    [{'partition': 'members'}, {'passes': [object()]}, {'backend': 'numpy'}, {'observables': ['x']}],
)
def test_compiler_check_rejects_invalid_settings(settings):
    """Test that `check` rejects a partition, passes, a backend, or observables of the wrong kind."""
    with pytest.raises(ValidationError):
        PassCompiler(**settings).prepare()


def test_compiler_rejects_a_target_that_is_not_a_system():
    """Test that a compiler prepares only on a system."""
    with pytest.raises(ValidationError, match='prepares on a system'):
        PassCompiler().prepare(object())


def test_general_compiler_names_the_compiler_of_a_operator_system():
    """Test that the general compiler reports an operator system with the compiler it needs, which compiles it."""
    system = System(elements=[QubitElement('q')])
    assert OperatorCompiler().prepare(system).report_.ok
    compiler = PassCompiler().prepare(system)
    assert not compiler.report_.ok
    with pytest.raises(ValidationError, match='compiles it with OperatorCompiler'):
        compiler.run()


def test_compiler_reports_an_observable_of_a_missing_system():
    """Test that an observable labelled with a nested system that does not exist is an issue of the report."""
    observable = Observable('village/size', lambda s: s.rabbits.x)
    report = PassCompiler(observables=[observable]).prepare(_growth()).report_
    assert [issue.code for issue in report.errors] == ['observable-path']


class Switch(Element):
    """Switch whose state `lit` is off or on."""

    def __init__(self, label: str, *, lit: Annotated[int, {0, 1}] = 0) -> None:
        super().__init__(label)
        self.lit = lit


def test_rate_of_a_value_of_a_finite_set_is_an_issue_of_its_domain():
    """Test that a rate law on a value allowed only 0 and 1 is rejected, since a rate moves it continuously."""
    turn = Interaction('turn', roles={'target': 'switch'}, laws=lambda s: [RateLaw(s.target.lit, 1.0)])
    system = System(elements=[Switch('switch')], interactions=[turn])
    with pytest.raises(ValidationError, match=r'changes switch\.lit continuously.*\{0, 1\}.*\[law-domain\]'):
        system.prepare()


def test_census_of_plain_elements_holds_an_observable_a_role_binds():
    """Test that a role bound to an observable of another nested system lets its law make a plain value a state."""
    copy = Interaction(
        'copy',
        roles={'target': 'ledger/tally', 'source': 'herds/largest'},
        laws=lambda s: [UpdateLaw(s.target.k, s.source.value)],
    )
    herds = System(
        'herds',
        elements=[Element('herd', values={'x': [1.0, 2.0, 3.0]}, size=3)],
        interactions=[Interaction('growth', roles={'target': 'herd'}, laws=lambda s: [RateLaw(s.target.x, 0.0)])],
    )
    ledger = System('ledger', elements=[Element('tally', values={'k': 0.0})])
    largest = Observable('herds/largest', group='herd', reduce=np.max, variable='x')
    system = System(systems=[ledger, herds], interactions=[copy]).prepare([largest])
    assert system.simulate(until=1, dt=1).get_values('ledger/tally.k')[-1] == pytest.approx(3.0)


def _nonlocal_field(label, law, space):
    """Return a field of the value `u` and the value `k` on a space, and the rate `law` of `u` it changes at.

    The law reads the namespace of the field, and the interaction that sets it is labelled `<label>_rate`.
    """
    field = Element(label, values={'u': 0.0, 'k': -1.0}, space=space)
    rate = _rule(f'{label}_rate', label, lambda s: [RateLaw(s.target.u, law(s.target))])
    return field, rate


def _nonlocal_report(*fields):
    """Return the report of a system of nonlocal fields with their rates."""
    elements = [field for field, _ in fields]
    interactions = [rate for _, rate in fields]
    return PassCompiler().prepare(System(elements=elements, interactions=interactions)).report_


def test_compiler_reports_a_negative_screening():
    """Test that a screening whose square is negative, or that reads a state, is a `screening` issue of the law."""
    for screening in (lambda s: sqrt(s.k), lambda s: s.u):
        field = _nonlocal_field(
            'm',
            lambda s, screening=screening: inverse_helmholtz(s.u, screening(s)),
            LatticeSpace(8),
        )
        report = _nonlocal_report(field)
        assert [(issue.code, issue.path) for issue in report.errors] == [('screening', 'm_rate')]
        assert 'screening' in report.errors[0].message


def test_compiler_reports_a_kernel_that_reads_a_state():
    """Test that a convolution kernel that reads a state is a `kernel-state` issue naming the state."""
    report = _nonlocal_report(_nonlocal_field('m', lambda s: convolve(s.u, s.u * displacement(0)), LatticeSpace(8)))
    assert [(issue.code, issue.path) for issue in report.errors] == [('kernel-state', 'm_rate')]
    assert 'u_m' in report.errors[0].message


def test_compiler_reports_a_two_point_kernel_that_reads_a_state():
    """Test that a two-point kernel that reads a state is a `kernel-state` issue naming the state."""
    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('z', 8, 0.0, 1.0)])
    field = _nonlocal_field('m', lambda s: integral_transform(s.u, s.u * integration_variable('z'), ('z',)), space)
    report = _nonlocal_report(field)
    assert [(issue.code, issue.path) for issue in report.errors] == [('kernel-state', 'm_rate')]
    assert 'u_m' in report.errors[0].message


def test_compiler_reports_a_two_point_integral_without_class_axes():
    """Test that an integral over class axes of a field on a lattice is a `law-space` issue."""
    field = _nonlocal_field('m', lambda s: integral_transform(s.u, integration_variable('z'), ('z',)), LatticeSpace(8))
    report = _nonlocal_report(field)
    assert {(issue.code, issue.path) for issue in report.errors} == {('law-space', 'm_rate')}
    assert any('two-point kernel' in issue.message for issue in report.errors)


def test_compiler_reports_a_cumulative_integral_on_a_network():
    """Test that a cumulative integral on a network is a `law-space` issue saying it needs a lattice space."""
    report = _nonlocal_report(
        _nonlocal_field('net', lambda s: cumulative_integral(s.u, 0), NetworkSpace(nx.path_graph(4))),
    )
    assert [(issue.code, issue.path) for issue in report.errors] == [('law-space', 'net_rate')]
    assert 'cumulative integral' in report.errors[0].message


def test_compiler_reports_a_local_operator_on_a_simplex():
    """Test that a Laplacian of a field on a simplex is a `law-space` issue asking for finite-volume fluxes."""
    space = SimplexSpace(LatticeSpace((1,)), [AxisSpec('x', 6, 0.0, 1.0), AxisSpec('y', 6, 0.0, 1.0)])
    report = _nonlocal_report(_nonlocal_field('m', lambda s: laplacian(s.u) * coordinate('x'), space))
    assert [(issue.code, issue.path) for issue in report.errors] == [('law-space', 'm_rate')]
    assert 'finite-volume' in report.errors[0].message


def test_compiler_reports_a_convolution_on_a_network():
    """Test that a convolution on a network is a `law-space` issue saying it needs a lattice space."""
    report = _nonlocal_report(
        _nonlocal_field('net', lambda s: convolve(s.u, displacement(0)), NetworkSpace(nx.path_graph(4))),
    )
    assert [(issue.code, issue.path) for issue in report.errors] == [('law-space', 'net_rate')]
    assert 'lattice space' in report.errors[0].message


def test_compiler_reports_spectral_operators_on_edges_that_are_not_periodic():
    """Test that a spectral inversion on a lattice with fixed edges is a `law-space` issue."""
    space = LatticeSpace(8, boundary='fixed', discretization='spectral')
    report = _nonlocal_report(_nonlocal_field('m', lambda s: inverse_helmholtz(s.u, 1), space))
    assert [issue.code for issue in report.errors] == ['law-space']


def test_compiler_reports_an_operand_over_two_spaces():
    """Test that an operand reading fields of two spaces is an `operator-spaces` issue."""
    first, _ = _nonlocal_field('a', lambda s: 0 * s.u, LatticeSpace(8))
    second, _ = _nonlocal_field('b', lambda s: 0 * s.u, LatticeSpace(8))
    cross = Interaction(
        'cross',
        roles={'first': 'a', 'second': 'b'},
        laws=lambda s: [RateLaw(s.first.u, inverse_helmholtz(s.first.u + s.second.u, 1))],
    )
    report = PassCompiler().prepare(System(elements=[first, second], interactions=[cross])).report_
    assert [issue.code for issue in report.errors] == ['operator-spaces']


def test_compiler_raises_every_nonlocal_issue_at_once():
    """Test that a convolution on a network and a negative screening raise one error that lists both laws."""
    network = _nonlocal_field('net', lambda s: convolve(s.u, displacement(0)), NetworkSpace(nx.path_graph(4)))
    lattice = _nonlocal_field('grid', lambda s: inverse_helmholtz(s.u, sqrt(s.k)), LatticeSpace(8))
    system = System(elements=[network[0], lattice[0]], interactions=[network[1], lattice[1]])
    compiler = PassCompiler().prepare(system)
    with pytest.raises(ValidationError, match=r'(?s)net_rate: .*law-space.*grid_rate: .*screening'):
        compiler.run()


class _TensorElement(Element):
    """Field with a symmetric traceless value `Q`."""

    def __init__(
        self,
        label: str,
        Q: Annotated[object, Tensor('traceless')] = 0.0,
        *,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.Q = Q


def _tensor_issues(rate):
    """Return the issues of a field whose tensor state has the given rate."""
    field = _TensorElement('film', space=LatticeSpace((4, 4)))
    relax = _rule('relax', 'film', lambda s: [RateLaw(s.target.Q, rate(s.target), label='relax')])
    return PassCompiler().prepare(System(elements=[field], interactions=[relax])).report_.issues


def test_prepare_reports_a_rate_with_a_trace_on_a_traceless_state():
    """A rate with a nonzero trace is reported with the law, the state, and the trace."""
    issues = _tensor_issues(lambda s: s.Q + identity_matrix(SR, 2))
    assert [issue.code for issue in issues] == ['tensor-symmetry']
    assert 'relax' in issues[0].message
    assert 'film.Q' in issues[0].message
    assert 'trace' in issues[0].message
    assert 'random component values' in issues[0].message


def test_prepare_accepts_the_entrywise_laplacian_of_a_traceless_state():
    """The Laplacian of a dependent entry, which Sage keeps unevaluated, keeps the symmetry."""
    assert _tensor_issues(lambda s: s.Q.apply_map(laplacian)) == ()


def test_prepare_accepts_a_rate_that_needs_simplification():
    """A rate whose trace vanishes only after an identity keeps the symmetry."""
    assert _tensor_issues(lambda s: (sin(s.Q[0, 0]) ** 2 + cos(s.Q[0, 0]) ** 2 - 1) * identity_matrix(SR, 2)) == ()


def test_prepare_reports_the_tensor_divergence_of_a_scalar():
    """The tensor divergence of a scalar state is reported with the utility and the state."""
    field = Element('film', values={'u': 0.0}, space=LatticeSpace((4, 4)))
    rate = _rule('rate', 'film', lambda s: [RateLaw(s.target.u, tensor_divergence(s.target.u)[0])])
    issues = PassCompiler().prepare(System(elements=[field], interactions=[rate])).report_.issues
    assert [issue.code for issue in issues] == ['law-domain']
    assert 'tensor_divergence' in issues[0].message
    assert 'u_film' in issues[0].message


def _cell_groups():
    """Return groups on the sites, the edges, and the hyperedges of the complex of a lattice."""
    space = CellComplexSpace(LatticeSpace((4, 4)))
    return [
        Element('sites', values={'u': 0.0}, space=space),
        Element('links', values={'a': 0.0}, space=space, on='edges'),
        Element('faces', values={'f': 0.0}, space=space, on='hyperedges'),
    ]


def _report_of(system, solver=None):
    """Return the report of a system, prepared with a solver if one is given."""
    compiler = PassCompiler() if solver is None else PassCompiler(solver=solver)
    return compiler.prepare(system).report_


def _cell_rate(label, target, state, rate, source=None):
    """Return the rate `rate(s)` of a value of a cell group, which may read the cells of a source group."""
    roles = {'target': target} if source is None else {'target': target, 'source': source}
    return Interaction(label, roles=roles, laws=lambda s: [RateLaw(getattr(s.target, state), rate(s))])


def _sampling(target, state, objective):
    """Return the objective `objective(s)` of a value of a group, which a sampling solver samples."""
    return Interaction(
        'sample',
        roles={'target': target},
        laws=lambda s: [ObjectiveLaw(objective(s), getattr(s.target, state))],
    )


def test_report_holds_one_degree_issue_per_law_with_its_own_path_and_fix():
    """Test that a codifferential of a site state and a sampling of a hyperedge group give two issues."""
    interactions = [
        _cell_rate('down', 'sites', 'u', lambda s: codifferential(s.target.u)),
        _sampling('faces', 'f', lambda s: s.target.f**2),
    ]
    solver = MetropolisSolver(moves={'faces.f': 'gaussian'}, schedule='colored')
    report = _report_of(System(elements=_cell_groups(), interactions=interactions), solver)
    issues = [issue for issue in report.issues if issue.code == 'law-degree']
    assert sorted(issue.path for issue in issues) == ['down', 'sample']
    assert 'exterior_derivative' in issues[0].fix
    assert 'edges' in issues[1].fix


def test_report_holds_a_degree_issue_for_an_exterior_derivative_of_the_highest_degree():
    """Test that an exterior derivative of a hyperedge state on a complex of dimension 2 is reported."""
    rate = _cell_rate('up', 'faces', 'f', lambda s: exterior_derivative(s.target.f))
    issues = _report_of(System(elements=_cell_groups(), interactions=[rate])).issues
    assert [issue.code for issue in issues if issue.path == 'up'] == ['law-degree']
    assert 'codifferential' in next(issue.fix for issue in issues if issue.path == 'up')


def test_report_holds_a_degree_issue_for_a_value_of_another_degree():
    """Test that a rate of a state from a value of another degree is reported, with or without an operation."""
    twice = _cell_rate('twice', 'links', 'a', lambda s: exterior_derivative(exterior_derivative(s.target.a)))
    bare = _cell_rate('bare', 'sites', 'u', lambda s: s.source.f, source='faces')
    wrong = _cell_rate('wrong', 'sites', 'u', lambda s: hodge_star(s.source.a), source='links')
    for interaction in (twice, bare, wrong):
        issues = _report_of(System(elements=_cell_groups(), interactions=[interaction])).issues
        assert [issue.path for issue in issues if issue.code == 'law-degree'] == [interaction.label]


def test_report_holds_no_degree_issue_for_operations_of_the_right_degree():
    """Test that a curl of a curl on links and a gradient of a site state are not reported."""
    interactions = [
        _cell_rate('curl', 'links', 'a', lambda s: codifferential(exterior_derivative(s.target.a))),
        _cell_rate('gradient', 'links', 'a', lambda s: exterior_derivative(s.source.u), source='sites'),
    ]
    issues = _report_of(System(elements=_cell_groups(), interactions=interactions)).issues
    assert not [issue for issue in issues if issue.code == 'law-degree']


def test_report_holds_no_degree_issue_for_a_group_mean_of_another_degree():
    """Test that a mean over the faces of an exterior derivative scales a link rate as a number of no degree."""
    rate = _cell_rate(
        'scaled',
        'links',
        'a',
        lambda s: mean(cos(exterior_derivative(s.target.a)), over=s.source) * s.target.a,
        source='faces',
    )
    issues = _report_of(System(elements=_cell_groups(), interactions=[rate])).issues
    assert not [issue for issue in issues if issue.code == 'law-degree']


def test_report_holds_a_degree_issue_for_a_misapplied_operation_inside_a_group_mean():
    """Test that a codifferential of a site state inside a group mean is still reported."""
    rate = _cell_rate('scaled', 'sites', 'u', lambda s: mean(codifferential(s.target.u)) * s.target.u)
    issues = _report_of(System(elements=_cell_groups(), interactions=[rate])).issues
    assert [issue.path for issue in issues if issue.code == 'law-degree'] == ['scaled']


def test_report_holds_a_space_issue_for_cofaces_outside_a_cell_complex():
    """Test that a coface reduction of a rate or of a sampling outside a cell complex is a law-space issue."""
    space = NetworkSpace(nx.cycle_graph(4))
    sites = Element('sites', values={'u': 0.0}, space=space)
    rate = _cell_rate('rate', 'sites', 'u', lambda s: neighbors(s.target.u, over=CofaceNeighborhood()))
    sampling = _sampling('sites', 'u', lambda s: neighbors(s.target.u, over=CofaceNeighborhood()))
    issues = _report_of(System(elements=[sites], interactions=[rate])).issues
    assert [issue.code for issue in issues if issue.path == 'rate'] == ['law-space']
    solver = MetropolisSolver(moves={'sites.u': 'gaussian'}, schedule='colored')
    issues = _report_of(System(elements=[sites], interactions=[sampling]), solver).issues
    assert [issue.code for issue in issues if issue.path == 'sample'] == ['law-space']


def test_report_holds_a_space_issue_for_the_exterior_calculus_outside_a_cell_complex():
    """Test that an exterior derivative of a state of a plain network is reported as a law-space issue."""
    sites = Element('sites', values={'u': 0.0}, space=NetworkSpace(nx.cycle_graph(4)))
    rate = _cell_rate('rate', 'sites', 'u', lambda s: exterior_derivative(s.target.u))
    issues = _report_of(System(elements=[sites], interactions=[rate])).issues
    assert [issue.code for issue in issues if issue.path == 'rate'] == ['law-space']


def test_compiler_reports_a_bad_setting_before_a_missing_system():
    """Test that preparing a compiler without a system names its invalid setting, not the missing system."""
    with pytest.raises(ValidationError, match='backend'):
        PassCompiler(backend='numba').prepare()


def _ecology(growth):
    """Return a system of one population under a growth interaction."""
    return System(elements=[PopulationElement('rabbits', x=10)], interactions=[growth])


def _hub_and_sites(interaction):
    """Return a hub and a group of five sites with values 0 to 4, coupled by an interaction."""
    return System(
        elements=[NodeElement('sites', x=np.arange(5.0), size=5), Element('hub', values={'y': 0.0})],
        interactions=[interaction],
    )


def _site_drive(site):
    """Return the growth of a hub at the rate of the value of one site."""
    return Interaction('link', roles={'hub': 'hub', 'site': site}, laws=lambda s: [RateLaw(s.hub.y, s.site.x)])


class SitesDriveInteraction(Interaction):
    """Growth of a hub at the rate of the summed values of several sites."""

    def __init__(self, label: str, *, hub: Role, sites: tuple[Role, ...]) -> None:
        super().__init__(label)
        self.hub = hub
        self.sites = sites

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the growth of the hub."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.hub.y, sum((item.x for item in context.sites), 0))]


def _compile_time(n_links):
    """Return the shortest of two compile times of a zero-range ring written with one interaction per link."""
    sites = Element('sites', values={'n': np.ones(n_links)}, space=LatticeSpace(n_links), on='sites')
    links = [
        ZeroRangeHopInteraction(f'hop_{index}', link=(f'sites[{index}]', f'sites[{(index + 1) % n_links}]'))
        for index in range(n_links)
    ]
    system = System(elements=[sites], interactions=links)
    durations = []
    for _ in range(2):
        start = time.perf_counter()
        PassCompiler(backend=NumpyBackend()).prepare(system).run()
        durations.append(time.perf_counter() - start)
    return min(durations)


class LennardJonesInteraction(Interaction):
    """Lennard-Jones pair potential between the members of two roles within the cutoff."""

    def __init__(
        self,
        label: str,
        *,
        first: Role,
        second: Role,
        cutoff: float = 1.0,
        epsilon: Annotated[float, RealSet.open(0, oo)] = 1.0,
        sigma: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label, within=RadiusNeighborhood(cutoff))
        self.first = first
        self.second = second
        self.cutoff = cutoff
        self.epsilon = epsilon
        self.sigma = sigma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the forces `-U'(r)` of `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)` on both members."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            separation = s.first.x - s.second.x
            r = sqrt(separation.dot_product(separation))
            distance = SR.var('r', domain='positive')
            energy = 4 * s.epsilon * ((s.sigma / distance) ** 12 - (s.sigma / distance) ** 6)
            force = -energy.diff(distance).subs({distance: r}) * separation / r
            self.laws_ = [RateLaw(s.first.x.diff(s.t, 2), force), RateLaw(s.second.x.diff(s.t, 2), -force)]


def _lennard_jones_gas(n, seed=0):
    """Return a gas of Lennard-Jones particles on a jittered lattice."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n] * 1.2
    space = ContinuousSpace(n_dim=2)
    values = {'x': grid + rng.normal(0, 0.05, (n, 2)), "x'": rng.normal(0, 0.5, (n, 2))}
    return System(
        elements=[BodyElement('gas', values=values, size=n, space=space)],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=space,
    )


class RecorderInteraction(Interaction):
    """Interaction that records the symbols it receives and adds no change."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Record the symbols and set a zero rate on the position."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            RECEIVED.update(x=context.target.x, m=context.target.m, t=context.t)
            self.laws_ = [RateLaw(context.target.x, 0 * context.target.x)]


@pytest.mark.parametrize(
    ('system', 'form'),
    [
        (_ecology(LogisticGrowthInteraction('growth', target='rabbits')), FormKind.ODE),
        (
            System(
                elements=[AssetElement('stock', price=100)],
                interactions=[GeometricBrownianMotionInteraction('gbm', target='stock')],
            ),
            FormKind.SDE,
        ),
        (
            System(
                elements=[AmountElement('I', x=5), AmountElement('R')],
                interactions=[RecoveryInteraction('rec', infected='I', recovered='R', stochastic=True)],
            ),
            FormKind.JUMP,
        ),
        (
            System(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')]),
            FormKind.DISCRETE,
        ),
        (System(elements=[PopulationElement('a')]), FormKind.STATIC),
    ],
)
def test_compiler_infers_form(system, form):
    """Test that the form follows from the law kinds."""
    assert PassCompiler().prepare(system).run().problem_.form == form


def test_compiler_mixed_kinds_raises_error():
    """Test that updates and rates in one system are rejected."""
    system = System(
        elements=[MarketElement('m'), PopulationElement('a')],
        interactions=[CobwebInteraction('c', target='m'), ExponentialGrowthInteraction('g', target='a')],
    )
    with pytest.raises(IncompatibleLawsError, match='RateLaw: a; UpdateLaw: m'):
        PassCompiler().prepare(system).run()


def test_compiler_names_nested_states_by_path():
    """Test that nested states are named by their path."""
    still = _rule('still', 'rabbits', lambda s: [RateLaw(s.target.x, 0)])
    system = System(systems=[System('meadow', elements=[AmountElement('rabbits', x=3)], interactions=[still])])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.states.names == ['meadow/rabbits.x']
    assert problem.initial_state.tolist() == [3.0]


def test_compiler_couples_subsystems():
    """Test that an interaction couples elements of two subsystems."""
    system = System(
        systems=[
            System('forest', elements=[PopulationElement('foxes', x=2)]),
            System('meadow', elements=[PopulationElement('rabbits', x=5)]),
        ],
        interactions=[PredationInteraction('hunt', predator='forest/foxes', prey='meadow/rabbits', a=1.0, b=0.5)],
    )
    problem = PassCompiler().prepare(system).run().problem_
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [5.0, -10.0]


def test_compiler_drift_takes_parameters():
    """Test that the drift reflects the parameter vector it receives."""
    system = _ecology(ExponentialGrowthInteraction('growth', target='rabbits', r=1.0))
    problem = PassCompiler().prepare(system).run().problem_
    x = problem.initial_state
    assert problem.drift(0.0, x, np.array([1.0])).tolist() == [10.0]
    assert problem.drift(0.0, x, np.array([2.0])).tolist() == [20.0]


def test_compiler_callable_law_is_evaluated():
    """Test that a callable rate is added to the drift and left out of the equations."""
    growth = CallableGrowthInteraction('growth', target='rabbits', r=3.0)
    system = System(elements=[HoldingElement('rabbits', n=10.0)], interactions=[growth])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [30.0]
    assert problem.has_callables
    assert problem.equations == {}


def test_compiler_rates_add_up():
    """Test that two forces on one particle add up."""
    space = ContinuousSpace(n_dim=1)
    system = System(
        elements=[BodyElement('p', x=[1.0], m=2.0)],
        interactions=[
            ConstantForceInteraction('push', target='p', F=4.0),
            HarmonicPotentialInteraction('trap', target='p', k=6.0),
        ],
        space=space,
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [0.0, -1.0]


def test_compiler_keeps_oscillator_equations():
    """Test that the equation of the velocity of an oscillator is `-k * (x - center) / m`."""
    space = ContinuousSpace(n_dim=1)
    system = System(
        elements=[BodyElement('p', x=[0.0])],
        interactions=[HarmonicPotentialInteraction('trap', target='p')],
        space=space,
    )
    problem = PassCompiler().prepare(system).run().problem_
    x, v = problem.state_symbols
    k = problem.symbols['trap.k']
    m = problem.symbols['p.m']
    center = problem.symbols['trap.center']
    assert problem.equations['p.x[0]'] == v
    assert (problem.equations['p.x_d1[0]'] + k * (x - center) / m).simplify_full() == 0


def test_compiler_vector_state_takes_space_dimension():
    """Test that vector states have one component per dimension."""
    rest = _rule('rest', 'p', lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0 * s.target.x)])
    system = System(elements=[BodyElement('p', x=[0.0, 0.0])], interactions=[rest], space=ContinuousSpace(n_dim=2))
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.states.component_names == ['p.x[0]', 'p.x[1]', 'p.x_d1[0]', 'p.x_d1[1]']


def test_compiler_vector_state_without_space_raises_error():
    """Test that a vector or tensor state needs a space."""
    relax = _rule('relax', 'film', lambda s: [RateLaw(s.target.Q, -s.target.Q)])
    with pytest.raises(SpaceError, match='must have a space'):
        PassCompiler().prepare(System(elements=[_TensorElement('film')], interactions=[relax])).run()


@pytest.mark.parametrize('m', [-2.0, 0.0, 'm'])
def test_compiler_value_outside_domain_raises_error(m):
    """Test that a parameter outside its domain is rejected."""
    space = ContinuousSpace(n_dim=1)
    system = System(elements=[BodyElement('p', x=[0.0], m=m)], space=space)
    with pytest.raises(ValidationError, match=r'p\.m'):
        PassCompiler().prepare(system).run()


def test_compiler_wrong_shape_raises_error():
    """Test that a vector value of the wrong size is rejected."""
    space = ContinuousSpace(n_dim=2)
    rest = _rule('rest', 'p', lambda s: [RateLaw(s.target.x, 0 * s.target.x)])
    system = System(elements=[BodyElement('p', x=[1, 2, 3])], interactions=[rest], space=space)
    with pytest.raises(ValidationError, match='shape'):
        PassCompiler().prepare(system).run()


def test_compiler_unknown_role_raises_error():
    """Test that a role must name an element."""
    system = _ecology(LogisticGrowthInteraction('growth', target='foxes'))
    with pytest.raises(RoleError, match="'target' to 'foxes'"):
        PassCompiler().prepare(system).run()


def test_compiler_duplicate_label_raises_error():
    """Test that siblings must have distinct labels."""
    system = System(elements=[PopulationElement('a'), PopulationElement('a')])
    with pytest.raises(LabelError, match="'a'"):
        PassCompiler().prepare(system).run()


def test_compiler_invalid_label_raises_error():
    """Test that labels must be words."""
    with pytest.raises(LabelError, match='not valid'):
        PassCompiler().prepare(System(elements=[PopulationElement('a.b')])).run()


def test_compiler_undeclared_symbol_raises_error():
    """Test that laws may only use declared symbols."""
    undeclared = _rule('bad', 'rabbits', lambda s: [RateLaw(s.target.x, symbol('k') * s.target.x)])
    with pytest.raises(ValidationError, match='symbols k'):
        PassCompiler().prepare(_ecology(undeclared)).run()


def test_compiler_double_update_raises_error():
    """Test that a state takes one update law."""
    twice = _rule('bad', 'm', lambda s: [UpdateLaw(s.target.price, 1), UpdateLaw(s.target.price, 2)])
    system = System(elements=[MarketElement('m')], interactions=[twice])
    with pytest.raises(ValidationError, match='second time'):
        PassCompiler().prepare(system).run()


def test_compiler_jump_stoichiometry():
    """Test that jump changes become rows of the stoichiometry matrix."""
    system = System(
        elements=[AmountElement('S', x=9), AmountElement('I', x=1)],
        interactions=[InfectionInteraction('inf', susceptible='S', infected='I', beta=0.5, stochastic=True)],
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.stoichiometry.tolist() == [[-1.0, 1.0]]
    assert problem.propensities(0.0, problem.initial_state, problem.initial_parameters).tolist() == [4.5]


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_compile_variadic_role_sums_any_number_of_inputs(backend):
    """Test that a variadic role binds a sequence of elements that laws read as a tuple of namespaces."""
    problem = PassCompiler(backend=backend()).prepare(_summed(['a', 'b', 'c'])).run().problem_
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(state, 'target.x') == 7.0
    assert problem.interactions[0].roles['inputs'] == ('a', 'b', 'c')


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_compile_variadic_member_reference_reads_one_member(backend):
    """Test that a member reference of a variadic role reads the states of that member of a group."""
    problem = PassCompiler(backend=backend()).prepare(_summed(['hub', 'nodes[3]'], weights=[0.0, 1.0])).run().problem_
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(state, 'target.x') == 3.0


@pytest.mark.parametrize(
    ('inputs', 'match'),
    [
        (['a', 'nodes[12]'], r"sum .*'nodes\[12\]'.*'inputs'.*10 members"),
        (['a', 'missing'], "'missing'"),
        ('a', 'Bind a list'),
    ],
)
def test_compile_variadic_role_with_wrong_entry_raises_error(inputs, match):
    """Test that a variadic entry naming a missing member or no element raises a role error."""
    with pytest.raises(RoleError, match=match):
        PassCompiler().prepare(_summed(inputs)).run()


def test_compile_variadic_group_entry_gives_symbolic_laws_no_states():
    """Test that a variadic entry naming a whole group, which binds its agents, gives symbolic laws no states."""
    with pytest.raises(RoleElementError, match="reads the state 'x'"):
        PassCompiler().prepare(_summed(['a', 'nodes'])).run()


def test_compile_plain_role_binds_one_group_member():
    """Test that a plain role bound to a member reference reads that member like a variadic entry does."""
    plain = PassCompiler().prepare(_hub_and_sites(_site_drive('sites[3]'))).run().problem_
    system = _hub_and_sites(SitesDriveInteraction('link', hub='hub', sites=('sites[3]',)))
    variadic = PassCompiler().prepare(system).run().problem_
    for problem in (plain, variadic):
        drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
        assert problem.states.read(drift, 'hub.y') == 3.0
    assert plain.interactions[0].roles['site'] == 'sites[3]'


def test_compile_plain_role_past_group_end_raises_error():
    """Test that a plain role bound past the end of a group names the interaction, role, reference, and size."""
    with pytest.raises(RoleError, match=r"link binds 'sites\[7\]' to its role 'site', but sites holds 5 members"):
        PassCompiler().prepare(_hub_and_sites(_site_drive('sites[7]'))).run()


def test_compile_per_entry_parameter_has_one_component_per_entry():
    """Test that a per-entry parameter is laid out with one addressable component per entry."""
    problem = PassCompiler().prepare(_summed(['a', 'b', 'c'], weights=[1.0, 2.0, 3.0])).run().problem_
    names = [name for name in problem.parameters.component_names if name.startswith('sum.')]
    assert names == ['sum.weights[0]', 'sum.weights[1]', 'sum.weights[2]']


@pytest.mark.benchmark
def test_compiler_time_grows_linearly_with_interactions():
    """Test that compiling 800 one-link interactions takes at most 2.5 times as long as compiling 400."""
    assert _compile_time(800) <= 2.5 * _compile_time(400)


def _gas(n, values=None, boundary='reflecting'):
    """Return a gas of `n` bodies at rest in a unit box with a boundary, whose positions move at their velocity."""
    space = ContinuousSpace(extent=(1.0, 1.0), boundary=boundary)
    gas = BodyElement('gas', values=values or {'x': np.full((n, 2), 0.5)}, size=n, space=space)
    rest = _rule('rest', 'gas', lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0 * s.target.x)])
    return System(elements=[gas], interactions=[rest], space=space)


@pytest.mark.benchmark
def test_compile_group_time_does_not_grow_with_size():
    """Test that compiling 100000 members takes no longer than 10 members within a factor of 2."""

    def compile_time(n):
        start = time.perf_counter()
        PassCompiler().prepare(_gas(n)).run()
        return time.perf_counter() - start

    compile_time(10)
    assert compile_time(100_000) <= 2 * compile_time(10) + 0.05


def test_compile_group_names_states_by_group_path():
    """Test that a group state is named by its group path with the group shape."""
    space = ContinuousSpace(n_dim=2)
    gas = BodyElement('gas', values={'x': np.zeros((500, 2))}, size=500, space=space)
    rest = _rule('rest', 'gas', lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0 * s.target.x)])
    problem = PassCompiler().prepare(System(elements=[gas], interactions=[rest], space=space)).run().problem_
    assert problem.states.get_entry('gas.x').shape == (500, 2)
    assert problem.equations['gas.x[i, 0]'] == problem.symbols['gas.x_d1'][0]


def test_compile_pair_forces_match_brute_force_sum():
    """Test that pair forces equal the sum over members closer than the cutoff."""
    problem = PassCompiler().prepare(_lennard_jones_gas(60)).run().problem_
    x, p = problem.initial_state, problem.initial_parameters
    positions = problem.states.read(x, 'gas.x')
    accelerations = problem.states.read(problem.drift(0.0, x, p), 'gas.x_d1')
    expected = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(len(positions)):
            separation = positions[i] - positions[j]
            r = np.linalg.norm(separation)
            if i != j and r < CUTOFF:
                expected[i] += 24 * (2 / r**13 - 1 / r**7) * separation / r
    assert np.allclose(accelerations, expected, rtol=1e-10, atol=1e-10)


def test_compile_pair_forces_between_two_groups():
    """Test that a pair interaction between two groups attracts both beyond the potential minimum."""
    space = ContinuousSpace(n_dim=2)
    system = System(
        elements=[
            BodyElement('a', values={'x': [[0.0, 0.0]]}, size=1, space=space),
            BodyElement('b', values={'x': [[1.5, 0.0]]}, size=1, space=space),
        ],
        interactions=[LennardJonesInteraction('lj', first='a', second='b', cutoff=CUTOFF)],
        space=space,
    )
    problem = PassCompiler().prepare(system).run().problem_
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    a, b = problem.states.read(drift, 'a.x_d1'), problem.states.read(drift, 'b.x_d1')
    assert a[0, 0] > 0
    assert np.allclose(a, -b)


def test_compile_pair_forces_use_minimum_image():
    """Test that periodic pair interactions use minimum image distances."""
    space = ContinuousSpace(extent=(10.0, 10.0), boundary='periodic')
    system = System(
        elements=[BodyElement('gas', values={'x': [[0.1, 0.5], [9.6, 0.5]]}, size=2, space=space)],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=space,
    )
    problem = PassCompiler().prepare(system).run().problem_
    accelerations = problem.states.read(
        problem.drift(0.0, problem.initial_state, problem.initial_parameters),
        'gas.x_d1',
    )
    assert accelerations[0, 0] > 0
    assert accelerations[1, 0] < 0


def test_compile_reflecting_box_keeps_ode_form():
    """Test that a reflecting box adds boundary laws without changing the form."""
    problem = PassCompiler().prepare(_gas(4)).run().problem_
    assert problem.form.name == 'ODE'
    assert problem.boundary is not None


@pytest.mark.parametrize(('name', 'wrapped'), [('x', 0.25), ('q', 1.25)])
def test_compile_periodic_box_wraps_places_named_after_its_axes(name, wrapped):
    """Test that a periodic box wraps a vector value named after its axes, a position, and no other."""
    space = ContinuousSpace(extent=(1.0, 1.0), boundary='periodic')
    gas = Element('gas', values={name: [[1.25, 0.5]]}, size=1, space=space)
    rest = _rule('rest', 'gas', lambda s: [RateLaw(getattr(s.target, name), 0.0 * getattr(s.target, name))])
    problem = PassCompiler().prepare(System(elements=[gas], interactions=[rest], space=space)).run().problem_
    state = problem.initial_state
    if problem.boundary is not None:
        state = problem.boundary(0.0, state, problem.initial_parameters)
    assert problem.states.read(state, f'gas.{name}')[0, 0] == pytest.approx(wrapped)


def test_compile_periodic_box_wraps_positions():
    """Test that a periodic box wraps a position that left the box."""
    problem = PassCompiler().prepare(_gas(1, {'x': [[1.25, 0.5]], "x'": [[1.0, 0.0]]}, 'periodic')).run().problem_
    state = problem.boundary(0.0, problem.initial_state, problem.initial_parameters)
    assert np.allclose(problem.states.read(state, 'gas.x'), [[0.25, 0.5]])
    assert np.allclose(problem.states.read(state, 'gas.x_d1'), [[1.0, 0.0]])


def _herds(make, x=(1.0, 2.0, 3.0)):
    """Return a group of herds whose amounts take the laws `make` returns from the namespace."""
    herds = AmountElement('herds', x=np.asarray(x), size=len(x))
    return System(elements=[herds], interactions=[_rule('law', 'herds', make)])


def test_compile_group_callable_rate():
    """Test that a callable law over a group returns one value per member."""
    problem = PassCompiler().prepare(_herds(lambda s: [RateLaw(s.target.x, lambda t, values: 2 * values['herds.x'])]))
    problem = problem.run().problem_
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [2.0, 4.0, 6.0]
    assert problem.has_callables


def test_compile_group_jumps_have_one_channel_per_member():
    """Test that a jump law over a group has one channel per member."""
    problem = PassCompiler().prepare(_herds(lambda s: [JumpLaw(1.0 * s.target.x, {s.target.x: 1})])).run().problem_
    assert problem.propensities(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, 3.0]
    assert problem.stoichiometry.tolist() == np.eye(3).tolist()


def test_compile_group_noise_has_one_process_per_member():
    """Test that a noise law over a group drives every member with its own process."""
    problem = PassCompiler().prepare(_herds(lambda s: [NoiseLaw(s.target.x, s.target.x)])).run().problem_
    assert problem.noise_rows.tolist() == [0, 1, 2]
    assert problem.diffusion(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, 3.0]


def test_compile_group_updates_every_member():
    """Test that a discrete update over a group updates every member."""
    system = _herds(lambda s: [UpdateLaw(s.target.x, s.target.x / 2)], x=(4.0, 8.0))
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.update(0.0, problem.initial_state, problem.initial_parameters).tolist() == [2.0, 4.0]
    assert problem.equations['herds.x[i]'] == problem.symbols['herds.x'] / 2


def test_compile_update_of_single_element_from_group_raises_error():
    """Test that an update cannot set a single element from a group."""
    update = Interaction(
        'sum',
        roles={'target': 'herds', 'total': 'total'},
        laws=lambda s: [UpdateLaw(s.total.x, s.target.x)],
    )
    system = System(
        elements=[AmountElement('herds', x=np.zeros(2), size=2), AmountElement('total')],
        interactions=[update],
    )
    with pytest.raises(ValidationError, match='single element from a group'):
        PassCompiler().prepare(system).run()


def test_compile_correction_twice_raises_error():
    """Test that a value takes one correction after each step."""
    wrap = _rule('wrap', 'gas', lambda s: [UpdateLaw(s.target.x[0], mod(s.target.x[0], 1))])
    space = ContinuousSpace(extent=(1.0,), boundary='reflecting')
    gas = BodyElement('gas', values={'x': np.full((2, 1), 0.5)}, size=2, space=space)
    rest = _rule('rest', 'gas', lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0 * s.target.x)])
    system = System(elements=[gas], interactions=[rest, wrap], space=space)
    with pytest.raises(ValidationError, match='second time'):
        PassCompiler().prepare(system).run()


def test_compiler_without_processes_keeps_identity_columns():
    """Test that noise laws that name no process get one process per amplitude and no correlated block."""
    walkers = Element('walkers', values={'x': np.zeros(3)}, size=3)
    walk = _rule('walk', 'walkers', lambda s: [NoiseLaw(s.target.x, 1.0), NoiseLaw(s.target.x, 2.0)])
    problem = PassCompiler().prepare(System(elements=[walkers], interactions=[walk])).run().problem_
    np.testing.assert_array_equal(problem.noise_columns, np.arange(6))
    assert problem.n_processes == 6
    assert problem.noise_blocks == ()


def test_compiler_shares_local_process():
    """Test that two laws of one element on one local process share one column."""
    problem = PassCompiler().prepare(pair_system()).run().problem_
    np.testing.assert_array_equal(problem.noise_columns, [0, 0])
    assert problem.n_processes == 1


def test_compiler_gives_members_their_own_local_process():
    """Test that a local process of a group law gives every member its own process."""
    problem = PassCompiler().prepare(pair_system(size=3)).run().problem_
    np.testing.assert_array_equal(problem.noise_columns, [0, 1, 2, 0, 1, 2])


def test_compiler_factors_covariance():
    """Test that a `WienerProcessSpec` with a covariance lowers to one block whose factor reproduces it."""
    problem = PassCompiler().prepare(_correlated([[1.0, 0.6], [0.6, 2.0]])).run().problem_
    (block,) = problem.noise_blocks
    assert (block.start, block.size) == (0, 2)
    np.testing.assert_allclose(block.factor @ block.factor.T, [[1.0, 0.6], [0.6, 2.0]])


def test_compiler_factors_singular_covariance():
    """Test that perfectly correlated processes compile with an eigenvalue factor."""
    (block,) = PassCompiler().prepare(_correlated([[1.0, 1.0], [1.0, 1.0]])).run().problem_.noise_blocks
    np.testing.assert_allclose(block.factor @ block.factor.T, np.ones((2, 2)), atol=1e-12)


def test_stratonovich_correction_covers_cross_terms():
    """Test that a shared Stratonovich process adds the cross terms `C G_kl dG_ij/dx_k` to the drift."""
    system = pair_system(conventions=('stratonovich', 'stratonovich'))
    problem = PassCompiler().prepare(system).run().problem_
    drift = problem.drift(0.0, np.asarray([2.0, 3.0]), problem.initial_parameters)
    np.testing.assert_allclose(drift, [0.5 * 0.5 * 0.5 * 2.0, 0.5 * 0.5 * 0.5 * 3.0])


def test_stratonovich_correction_of_several_laws_adds_up():
    """Test that two Stratonovich laws with independent processes on one state compile and add their corrections."""
    twice = _rule(
        'noise',
        'twice',
        lambda s: [
            NoiseLaw(s.target.x, s.target.x, convention='stratonovich'),
            NoiseLaw(s.target.x, 2 * s.target.x, convention='stratonovich'),
        ],
    )
    system = System(elements=[Element('twice', values={'x': 1.0})], interactions=[twice])
    problem = PassCompiler().prepare(system).run().problem_
    drift = problem.drift(0.0, np.asarray([2.0]), problem.initial_parameters)
    assert drift[0] == pytest.approx(0.5 * 2.0 + 0.5 * 4 * 2.0)


def test_stratonovich_callable_amplitude_raises_an_error_naming_the_drift_correction():
    """Test that a Stratonovich noise law with a callable amplitude raises an error instead of running as Itô."""
    noise = _rule(
        'noise',
        'noisy',
        lambda s: [NoiseLaw(s.target.x, lambda t, values: values['noisy.x'], convention='stratonovich')],
    )
    system = System(elements=[Element('noisy', values={'x': 1.0})], interactions=[noise])
    with pytest.raises(ValidationError, match='drift correction'):
        PassCompiler().prepare(system).run()


def test_components_receive_sage_symbols_and_vectors():
    """Test that values are Sage functions of the time with their domains and vector values are Sage vectors."""
    space = ContinuousSpace(n_dim=3)
    system = System(
        elements=[BodyElement('p', x=[0.0, 0.0, 0.0])],
        interactions=[RecorderInteraction('record', target='p')],
        space=space,
    )
    PassCompiler().prepare(system).run()
    assert isinstance(RECEIVED['x'], FreeModuleElement)
    assert len(RECEIVED['x']) == 3
    assert all(isinstance(component, Expression) for component in RECEIVED['x'])
    assert all(component.operands() == [RECEIVED['t']] for component in RECEIVED['x'])
    assert domain_of(RECEIVED['m']) is RR


@pytest.mark.benchmark
def test_small_system_drift_is_fast():
    """Test that the NumPy drift of 20 logistic populations takes at most 250 microseconds and is correct."""
    system = System(
        elements=[PopulationElement(f'p{index}', x=1.0 + index) for index in range(20)],
        interactions=[LogisticGrowthInteraction(f'g{index}', target=f'p{index}', r=1.0, K=100) for index in range(20)],
    )
    problem = PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_
    x, p = problem.initial_state, problem.initial_parameters
    expected = x * (1 - x / 100)
    assert problem.drift(0.0, x, p) == pytest.approx(expected)
    repeats = 500
    start = time.perf_counter()
    for _ in range(repeats):
        problem.drift(0.0, x, p)
    assert (time.perf_counter() - start) / repeats <= 250e-6


def test_pass_compiler_get_ordered():
    """Test `PassCompiler._get_ordered` on a small case."""

    class LowerPass(BasePass):
        def _get_reads(self):
            return ('layout',)

        def _get_writes(self):
            return ('lowered',)

    class DeclarePass(BasePass):
        def _get_reads(self):
            return ('nodes',)

        def _get_writes(self):
            return ('layout',)

    ordered = PassCompiler._get_ordered([LowerPass(), DeclarePass()], present=['nodes'])
    assert [type(item).__name__ for item in ordered] == ['DeclarePass', 'LowerPass']


def test_passes_are_ordered_produced_refined_then_read():
    """Test that a producer runs before the refiners of its field, and those before its readers."""
    passes = [
        _pass('Read', ('layout',), ('kernels',)),
        _pass('Refine', ('layout',), ('layout',)),
        _pass('Produce', ('nodes',), ('layout',)),
    ]
    assert _names(PassCompiler._get_ordered(passes, present=['nodes'])) == ['Produce', 'Refine', 'Read']


def test_passes_keep_the_given_order_of_refiners():
    """Test that refiners of one field run in the order they are given."""
    passes = [
        _pass('Produce', (), ('layout',)),
        _pass('SecondElement', ('layout',), ('layout',)),
        _pass('FirstElement', ('layout',), ('layout',)),
    ]
    assert _names(PassCompiler._get_ordered(passes)) == ['Produce', 'SecondElement', 'FirstElement']


def test_ordering_names_a_missing_input():
    """Test that a field no pass produces and the compilation lacks names the pass that reads it."""
    with pytest.raises(PipelineError, match="Lower reads the field 'layout'"):
        PassCompiler._get_ordered([_pass('Lower', ('layout',), ('lowered',))], present=['nodes'])


def test_ordering_accepts_a_field_the_compilation_holds():
    """Test that a field the compilation holds needs no producer."""
    assert _names(PassCompiler._get_ordered([_pass('Lower', ('layout',), ('lowered',))], present=['layout'])) == [
        'Lower',
    ]


def test_ordering_rejects_two_producers():
    """Test that two passes producing one field are rejected."""
    with pytest.raises(PipelineError, match='both produce'):
        PassCompiler._get_ordered([_pass('One', (), ('layout',)), _pass('Two', (), ('layout',))])


def test_ordering_rejects_a_cycle():
    """Test that passes reading each other's fields are rejected."""
    with pytest.raises(PipelineError, match='cycle'):
        PassCompiler._get_ordered([_pass('One', ('b',), ('a',)), _pass('Two', ('a',), ('b',))])


def test_passes_run_only_what_a_field_needs():
    """Test that running until a field runs the passes that produce it and their inputs, once each."""
    log = []
    passes = [
        _pass('Problem', ('lowered',), ('problem',), log),
        _pass('Lower', ('layout',), ('lowered',), log),
        _pass('Layout', ('system',), ('layout',), log),
        _pass('OtherElement', ('system',), ('census',), log),
    ]
    compiler = PassCompiler(passes=passes)
    compiler.passes_ = tuple(passes)
    compilation = compiler._get_new_compilation(System())
    compiler._run_passes(compilation, until='lowered')
    assert log == ['Layout', 'Lower']
    compiler._run_passes(compilation)
    assert log == ['Layout', 'Lower', 'Problem']


@pytest.mark.skip(
    reason=(
        'Engine gap: the general compiler flattens a `System` twice, once for the census that infers its '
        'states and once in its passes.'
    ),
)
def test_compiler_flattens_the_system_once(monkeypatch):
    """Test that validating and compiling a system whose values are declared flattens its tree once."""
    calls = []
    flatten = FlattenPass._flatten_system

    def counted(*args, **kwargs):
        calls.append(args[0])
        return flatten(*args, **kwargs)

    system = LotkaVolterraSystem().prepare().system_
    monkeypatch.setattr(FlattenPass, '_flatten_system', staticmethod(counted))
    PassCompiler(backend=NumpyBackend()).prepare(system).run()
    assert len(calls) == 1


def test_valid_system_has_no_issues():
    """Test that a system of stocks with growth validates without issues."""
    system = System(elements=[StockElement('rabbits')], interactions=[GrowthInteraction('growth', target='rabbits')])
    report = PassCompiler().prepare(system).report_
    assert report.ok
    assert report.issues == ()


def test_several_errors_are_reported_at_once():
    """Test that an unbound role and a tensor state without a space are both reported."""
    relax = _rule('relax', 'film', lambda s: [RateLaw(s.target.Q, -s.target.Q)])
    system = System(
        elements=[StockElement('rabbits'), _TensorElement('film')],
        interactions=[GrowthInteraction('growth', target='foxes'), relax],
    )
    report = PassCompiler().prepare(system).report_
    assert sorted(_codes(report)) == ['role', 'space-missing']
    assert {issue.path for issue in report.errors} == {'growth', 'film'}
    assert all(issue.fix for issue in report.issues)


def _count_steps(agent, system):
    """Count one step of an agent."""
    agent.steps += 1


def test_unsupported_solver_lists_capable_solvers():
    """Test that agents validated with RK4 report the unsupported components and capable solvers."""
    ants = Element('ants', values={'steps': np.zeros(3)}, size=3)
    walk = _rule('walk', 'ants', lambda s: [UpdateLaw(s.target, _count_steps)])
    report = PassCompiler(solver=RK4Solver()).prepare(System(elements=[ants], interactions=[walk])).report_
    issue = next(issue for issue in report.errors if 'laws' in issue.message)
    assert issue.code == 'solver-unsupported'
    assert 'UpdateLaw' in issue.message
    assert 'UpdateLaw' in issue.fix


def test_mixed_laws_suggest_partitions():
    """Test that rates and updates in one system without children are reported with both components."""
    system = System(
        elements=[StockElement('rabbits'), CounterElement('clock')],
        interactions=[GrowthInteraction('growth', target='rabbits'), _tick('tick', 'clock')],
    )
    (issue,) = PassCompiler().prepare(system).report_.errors
    assert issue.code == 'laws-mixed'
    assert 'RateLaw: rabbits' in issue.message
    assert 'UpdateLaw: clock' in issue.message
    assert 'nested system' in issue.fix


def test_law_of_a_class_with_no_form_names_its_class():
    """Test that a law of a class with no form raises the unregistered law error, not a mix of laws."""
    decay = _rule('decay', 'rabbits', lambda s: [ScaledRateLaw(s.target.n, -s.target.n)])
    system = System(
        elements=[StockElement('rabbits'), CounterElement('clock')],
        interactions=[decay, _tick('tick', 'clock')],
    )
    (issue,) = PassCompiler().prepare(system).report_.errors
    assert (issue.code, issue.error) == ('law-unregistered', UnregisteredLawError)
    with pytest.raises(UnregisteredLawError, match=r'rabbits: The law class ScaledRateLaw has no form'):
        PassCompiler().prepare(system).run()


def test_mixed_laws_in_nested_systems_are_valid():
    """Test that rates and updates in sibling nested systems validate as coupled partitions."""
    system = System(
        systems=[
            System(
                'meadow',
                elements=[StockElement('rabbits')],
                interactions=[GrowthInteraction('growth', target='rabbits')],
            ),
            System('clock', elements=[CounterElement('counter')], interactions=[_tick('tick', 'counter')]),
        ],
    )
    assert PassCompiler().prepare(system).report_.ok


def test_validate_control_outside_bounds():
    """Test that a control starting outside its bounds is one error at the control, naming its bounds."""
    system = System(elements=[PlantElement('plant', u=2.0)], interactions=[_plant()])
    (issue,) = PassCompiler(solver=PontryaginSolver()).prepare(system).report_.errors
    assert issue.path == 'plant.u'
    assert issue.code == 'parameter-bounds'
    assert re.search(r'\(0(\.0)?, 1(\.0)?\)', issue.fix)


def test_validate_controls_with_an_optimal_control_solver():
    """Test that a system with bounded controls is valid for an optimal control solver."""
    system = System(elements=[PlantElement('plant')], interactions=[_plant()])
    assert PassCompiler(solver=PontryaginSolver()).prepare(system).report_.ok


def test_validate_noise_with_an_optimal_control_solver():
    """Test that a system with a noise law validated with Pontryagin reports that the solver lacks noise laws."""
    system = System(elements=[PlantElement('plant')], interactions=[_plant(noisy=True)])
    issues = PassCompiler(solver=PontryaginSolver()).prepare(system).report_.errors
    issue = next(issue for issue in issues if issue.code == 'solver-unsupported')
    assert 'PontryaginSolver does not support the laws NoiseLaw' in issue.message


class SpinElement(Element):
    """Spins at the sites of a space, at the temperature `T`."""

    def __init__(
        self,
        label: str,
        *,
        spin: int = 1,
        T: float = 2.0,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label, space=space, on=on)
        self.spin = spin
        self.T = T


def _ferromagnet(label: str = 'ferromagnet') -> Interaction:
    """Return the ferromagnetic energy of a spin and its neighbours along the generators of a group."""

    def laws(s):
        """Return the energy over the temperature."""
        memory = MemoryNeighborhood(('a', 'a^-1', 'b', 'b^-1'))
        spin = s.target.spin
        energy = -spin * sum(neighbors(spin, memory, reduce='each'))
        return [ObjectiveLaw(energy / s.target.T, spin)]

    return _rule(label, 'spins', laws)


def test_group_space_reports_the_checkerboard_schedule():
    """Test that a sample with the checkerboard schedule on a group space is one issue naming the other schedules."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2)
    system = System(elements=[SpinElement('spins', space=space)], interactions=[_ferromagnet()])
    report = PassCompiler(solver=MetropolisSolver(rule='glauber', moves={'spins.spin': 'flip'})).prepare(system).report_
    (issue,) = [issue for issue in report.issues if issue.code == 'schedule-unsupported']
    assert issue.path == 'ferromagnet'
    assert "'synchronous'" in issue.fix
    assert "'random-sequential'" in issue.fix


def test_group_space_draws_an_image_member_as_the_group_space():
    """Test that a member drawn as an image is drawn as the Cayley graph of a group space, with no visual issue."""

    class ImageSpinElement(SpinElement):
        """Spin drawn as an image."""

        visual = VisualSpec('image')

    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2)
    compiler = PassCompiler().prepare(System(elements=[ImageSpinElement('spins', space=space)]))
    assert not [issue for issue in compiler.report_.issues if 'visual' in issue.code]
    assert compiler.run().problem_.elements[0].visual.kind is CayleyVisual


def test_group_space_reports_a_ball_above_the_site_limit():
    """Test that a ball of the free group of rank 3 with more sites than the limit is reported."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b', 'c']), ball=12, max_sites=100)
    issues = PassCompiler().prepare(System(space=space)).report_.issues
    assert [issue.code for issue in issues] == ['space-too-large']
    assert 'more than 100 sites' in issues[0].message


def test_lattice_with_a_bad_boundary_is_reported():
    """Test that a lattice whose boundary is not supported is a space-invalid error that lists the boundaries."""
    space = LatticeSpace((4, 4), boundary='spherical')
    (issue,) = PassCompiler().prepare(System(elements=[StockElement('rabbits')], space=space)).report_.errors
    assert issue.code == 'space-invalid'
    assert "'spherical'" in issue.message
    assert 'periodic, fixed, reflecting' in issue.message


def test_invalid_space_among_other_errors_is_reported():
    """Test that an invalid lattice and an unbound role are both reported, and compiling raises the validation error."""
    system = System(
        elements=[StockElement('rabbits')],
        interactions=[GrowthInteraction('growth', target='hares')],
        space=LatticeSpace((4, 4), boundary='spherical'),
    )
    assert {issue.code for issue in PassCompiler().prepare(system).report_.errors} == {'space-invalid', 'role'}
    with pytest.raises(ValidationError, match='space-invalid'):
        PassCompiler().prepare(system).run()


def test_validation_keeps_the_planned_compilation():
    """Test that validation returns the compilation state planned once, which holds the nodes and the plan."""
    system = System(elements=[StockElement('rabbits')], interactions=[GrowthInteraction('growth', target='rabbits')])
    compilation = PassCompiler().prepare(system).compilation_
    assert [node.path for node in compilation.nodes] == ['rabbits', 'growth']
    assert [partition.name for partition in compilation.plan.partitions] == [system.label]


def test_validate_reports_mixed_conventions():
    """Test that laws of different conventions on one process are reported with both laws and the process."""
    report = PassCompiler().prepare(pair_system(conventions=('ito', 'stratonovich'))).report_
    (issue,) = [issue for issue in report.issues if issue.code == 'noise-convention-mixed']
    assert 'ito NoiseLaw of noise' in issue.message
    assert 'stratonovich NoiseLaw of noise' in issue.message
    assert "'common'" in issue.message
    assert 'one convention' in issue.fix


def test_validate_reports_correction_out_of_scope():
    """Test that a Stratonovich law reading a state another component drives on its process is reported."""
    system = System(
        elements=[HoldingElement('a'), HoldingElement('b', n=1.0)],
        interactions=[
            CoupledInteraction('coupled', target='a', other='b', noise=WienerProcessSpec('market')),
            SingleInteraction('single', target='b', noise=WienerProcessSpec('market')),
        ],
    )
    report = PassCompiler().prepare(system).report_
    (issue,) = [issue for issue in report.issues if issue.code == 'noise-correction-scope']
    assert 'NoiseLaw of coupled' in issue.message
    assert 'NoiseLaw of single' in issue.message
    assert 'Itô' in issue.fix


def test_validate_reports_invalid_covariance():
    """Test that a covariance with a negative eigenvalue is reported with the element and the eigenvalue."""
    (issue,) = PassCompiler().prepare(_correlated([[1.0, 2.0], [2.0, 1.0]])).report_.issues
    assert issue.code == 'wiener-covariance'
    assert issue.path == 'market'
    assert '-1' in issue.message


@pytest.mark.parametrize(
    'covariance',
    [[[1.0, 0.0, 0.0]], [[1.0, 0.5], [0.0, 1.0]]],
)
def test_validate_reports_covariance_shape_and_symmetry(covariance):
    """Test that a covariance that is not square of the dimension or not symmetric is reported."""
    report = PassCompiler().prepare(_correlated(covariance)).report_
    assert [issue.code for issue in report.issues] == ['wiener-covariance']


def test_validate_reports_handle_as_value():
    """Test that a Wiener handle used inside a rate law is reported with the law."""
    system = System(
        elements=[HoldingElement('a')],
        interactions=[MisuseInteraction('misuse', target='a', noise=WienerProcessSpec('market'))],
    )
    report = PassCompiler().prepare(system).report_
    (issue,) = [issue for issue in report.issues if issue.code == 'wiener-handle']
    assert 'RateLaw of misuse' in issue.message
    assert 'only the process of a noise law' in issue.message


def test_validate_reports_milstein_with_shared_process():
    """Test that Milstein with a shared process gives one error that names the law and suggests other solvers."""
    report = PassCompiler(solver=MilsteinSolver()).prepare(pair_system()).report_
    (issue,) = report.issues
    assert issue.code == 'noise-unsupported'
    assert 'NoiseLaw of noise' in issue.message
    assert 'MilsteinSolver' in issue.message
    assert 'shared noise' in issue.fix


def test_validate_reports_lammps_with_shared_process():
    """Test that the LAMMPS backend with a shared process is reported with the backends that support it."""
    issues = PassCompiler(solver=LammpsSolver()).prepare(pair_system()).report_.issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-unsupported']
    assert 'NoiseLaw of noise' in issue.message
    assert 'LammpsSolver' in issue.message


def test_validate_accepts_correlated_noise_with_automatic_solver():
    """Test that the automatic SDE solver declares correlated noise, so validation accepts it."""
    assert PassCompiler().prepare(_correlated([[1.0, 0.5], [0.5, 1.0]])).report_.ok
    assert NoiseKind.CORRELATED in DEFAULT_SOLVERS[FormKind.SDE]._get_support().noise


def test_observable_among_the_laws_is_rejected():
    """Test that an observable among the laws is an error that names the component and the laws it takes."""
    mixed = _rule('mixed', 'tank', lambda s: [RateLaw(s.target.a, -s.target.a), Observable('level', s.target.a)])
    system = System(elements=[Element('tank', values={'a': 1.0})], interactions=[mixed])
    (issue,) = PassCompiler().prepare(system).report_.errors
    assert issue.path == 'mixed'
    assert 'of mixed must return a list of' in issue.message
    with pytest.raises(ValidationError, match='mixed'):
        PassCompiler().prepare(system).run()


def test_neighbor_interaction_without_neighbours_takes_no_laws_per_edge():
    """Test that an interaction over neighbours of members on no lattice or network binds but has no laws per edge."""
    assert _bindings(_spread(), None) == []
    nodes = NodeElement('nodes', x=np.zeros(4), size=4)
    system = System(elements=[nodes], interactions=[_spread()])
    with pytest.raises(ValidationError, match='laws per edge'):
        PassCompiler().prepare(system).run()


def test_neighbor_interaction_on_a_missing_layer_is_reported():
    """Test that an interaction over neighbours on a layer the network does not have names the layers it has."""
    (issue,) = _bindings(_spread('work'), NetworkSpace(nx.cycle_graph(4)))
    assert "the layer 'work'" in issue.message
    assert 'LayerSpec' in issue.fix


def test_neighbor_interaction_on_a_network_is_valid():
    """Test that an interaction over neighbours of members on a network has no binding issue."""
    assert _bindings(_spread(), NetworkSpace(nx.cycle_graph(4))) == []


def test_hyperedge_interaction_without_hyperedges_is_reported():
    """Test that an interaction over hyperedges of members on a network without hyperedges is reported."""
    (issue,) = _bindings(_consensus(), NetworkSpace(nx.cycle_graph(4)))
    assert 'a network space with hyperedges' in issue.message


def test_hyperedge_interaction_on_hyperedges_is_valid():
    """Test that an interaction over hyperedges of members on a network with hyperedges has no binding issue."""
    space = NetworkSpace(nx.cycle_graph(4), hyperedges=[(0, 1, 2), (1, 3)])
    assert _bindings(_consensus(), space) == []


def test_validate_shaped_parameter_on_an_element():
    """Test that a parameter shaped by a role bound to a single element is one issue at the parameter."""
    report = PassCompiler().prepare(_weighted(np.ones((4, 2)), [1.0, 2.0], source='single')).report_
    (issue,) = report.errors
    assert (issue.code, issue.path) == ('parameter-shape', 'input')
    assert 'input.W' in issue.message
    assert 'group' in issue.fix


def test_role_element_reports_a_missing_state_a_role_declares():
    """Test that a read of a value the element lacks is reported with its path, role, value, and fix."""
    growth = _rule('growth', 'tank', lambda s: [RateLaw(s.target.n, 1.0)])
    issues = _role_issues(System(elements=[TankElement('tank')], interactions=[growth]))
    assert len(issues) == 1
    assert issues[0].path == 'growth'
    assert "'target'" in issues[0].message
    assert "'n'" in issues[0].message
    assert 'level' in issues[0].message
    assert 'get_requirements()' in issues[0].fix


def test_role_element_reports_a_missing_state_no_role_declares():
    """Test that an undeclared read of a missing state gets the same issue, and no `AttributeError` reaches the user."""
    system = System(elements=[TankElement('tank')], interactions=[_reading()])
    issues = _role_issues(system)
    assert len(issues) == 1
    assert "'target'" in issues[0].message
    assert "'n'" in issues[0].message
    assert 'level' in issues[0].message
    with pytest.raises(RoleElementError, match=r"reads the state 'n'.*\[role-element\]"):
        PassCompiler().prepare(system).run()


def test_role_element_names_a_variadic_role():
    """Test that a missing state of an element of a variadic role names this role."""

    class ReadingSourcesInteraction(Interaction):
        """Interaction that reads a value of the elements of a variadic role."""

        def __init__(self, label: str, *, target: Role, sources: tuple[Role, ...]) -> None:
            super().__init__(label)
            self.target = target
            self.sources = sources

        def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
            """Set a rate of the target that reads the value `missing` of its first source."""
            super()._prepare(context)
            if isinstance(context, SimpleNamespace):
                self.laws_ = [RateLaw(context.target.x, context.sources[0].missing)]

    system = System(
        elements=[Element('first', values={'x': 1.0}), TankElement('tank')],
        interactions=[ReadingSourcesInteraction('read', target='first', sources=('tank',))],
    )
    issues = _role_issues(system)
    assert len(issues) == 1
    assert "'sources'" in issues[0].message
    assert "'missing'" in issues[0].message


def test_role_element_comes_with_the_other_problems():
    """Test that a role issue is one problem of the report, next to a problem of another component."""
    system = System(
        elements=[TankElement('tank')],
        interactions=[
            _reading(),
            _rule('growth', 'tank', lambda s: [RateLaw(s.target.n, 1.0)]),
            _rule('unbound', 'nowhere', lambda s: [RateLaw(s.target.n, 1.0)]),
        ],
    )
    issues = PassCompiler().prepare(system).report_.issues
    assert {issue.path for issue in issues if issue.code == 'role-element'} == {'read', 'growth'}
    assert any(issue.path == 'unbound' and issue.code == 'role' for issue in issues)


def test_role_element_keeps_a_missing_name_of_the_namespace():
    """Test that reading a name the namespace itself lacks is not reported as a state of a role."""
    wrong = _rule('read', 'tank', lambda s: [RateLaw(s.target.level, s.undeclared)])
    system = System(elements=[TankElement('tank')], interactions=[wrong])
    with pytest.raises(AttributeError, match='undeclared'):
        PassCompiler().prepare(system)


@pytest.mark.parametrize('error', [ValueError('own error'), AttributeError('own error'), KeyError('own error')])
def test_role_element_keeps_the_other_errors_of_a_law(error):
    """Test that an error of a law that is not a missing attribute of a role reaches the user unchanged."""

    def failing(s):
        """Raise the error of the test."""
        raise error

    system = System(elements=[TankElement('tank')], interactions=[_rule('read', 'tank', failing)])
    with pytest.raises(type(error), match='own error') as caught:
        PassCompiler().prepare(system)
    assert caught.value is error


def test_role_element_leaves_a_valid_binding_alone():
    """Test that an element that carries what a role reads reports no role issue."""
    growth = _rule('growth', 'tank', lambda s: [RateLaw(s.target.level, 1.0)])
    assert _role_issues(System(elements=[TankElement('tank')], interactions=[growth])) == []


def test_role_element_reports_a_state_that_a_function_reads():
    """Test that a function that sets the laws and reads a missing state gets the same issue."""
    growth = _rule('growth', 'tank', lambda s: [RateLaw(s.target.level, s.target.missing)])
    issues = _role_issues(System(elements=[TankElement('tank')], interactions=[growth]))
    assert len(issues) == 1
    assert "'target'" in issues[0].message
    assert "'missing'" in issues[0].message


def _hypergraph_variance() -> System:
    """Return nodes whose rate is the variance over the members of their hyperedges, which no hyperedge writes out."""
    space = NetworkSpace(nx.complete_graph(4), hyperedges=[(0, 1, 2), (1, 2, 3)])
    rule = Interaction(
        'spread',
        among='nodes',
        along='hyperedges',
        laws=lambda s: [RateLaw(s.member.x, variance(s.members.x, over=s.members))],
    )
    return System(elements=[Element('nodes', values={'x': np.zeros(4)}, space=space)], interactions=[rule])


def test_compiler_reports_the_failure_of_the_probe_in_place_of_the_later_one():
    """Test that a component that fails while its states are inferred is reported with that first failure."""
    report = PassCompiler().prepare(_hypergraph_variance()).report_
    assert [issue.path for issue in report.issues] == ['spread']
    assert 'total, mean, count, maximum, or minimum' in report.issues[0].message


def _pond():
    """Return the algae of Lindenmayer as a system of one word, whose length is a Fibonacci number."""
    growth = RewritingInteraction('growth', target='algae', laws=[ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')])
    return System('pond', elements=[WordElement('algae', axiom='A')], interactions=[growth])


@pytest.mark.parametrize(
    ('laws', 'expected'),
    [
        (lambda s: [RateLaw(s.target.y, -s.target.y)], np.exp(-np.arange(5.0))),
        (lambda s: [UpdateLaw(s.target.y, s.target.y + 1)], np.arange(1.0, 6.0)),
    ],
)
def test_nested_rewriting_system_runs_as_a_partition_of_its_own(laws, expected):
    """Test that a system of words nested beside a continuous or a discrete element keeps its own time and solver."""
    rule = Interaction('change', roles={'target': 'probe'}, laws=laws)
    system = System(elements=[Element('probe', values={'y': 1.0})], interactions=[rule], systems=[_pond()])
    trajectory = system.prepare().simulate(until=4, dt=1)
    assert trajectory.get_values('pond/algae.size').tolist() == [1.0, 2.0, 3.0, 5.0, 8.0]
    np.testing.assert_allclose(trajectory.get_values('probe.y'), expected, rtol=1e-5)


def test_pass_compiler_compiles_a_system_of_words_by_the_compiler_its_words_name():
    """Test that the pass compiler hands a system of words to the rewriting compiler and keeps its partitions apart."""
    alone = PassCompiler().prepare(_pond()).run().problem_
    assert str(alone.form) == 'rewriting system'
    decay = Interaction('decay', roles={'target': 'probe'}, laws=lambda s: [RateLaw(s.target.y, -s.target.y)])
    beside = System(elements=[Element('probe', values={'y': 1.0})], interactions=[decay], systems=[_pond()])
    problem = PassCompiler().prepare(beside).run().problem_
    assert problem.form == FormKind.COUPLED
    assert [partition.name for partition in problem.partitions] == ['system', 'pond']
    assert problem.states.names == ['probe.y', 'pond/algae.A', 'pond/algae.B', 'pond/algae.size']
