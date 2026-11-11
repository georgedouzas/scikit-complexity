"""Test the general compiler: its settings, its report on a system, and the problem it runs to."""

import re
import time
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_gap import libgap
from sage.all__sagemath_symbolics import RR, SR, ZZ, Expression, cos, identity_matrix, oo, sin, sqrt
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.interactions import (
    HyperedgeInteraction,
    Interaction,
    NeighborInteraction,
    PairInteraction,
    RateInteraction,
    SamplingInteraction,
)
from skcomplex.core.components.laws import BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, RuleLaw, SampleLaw, UpdateLaw
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
)
from skcomplex.core.components.spaces import (
    CayleySpace,
    CellComplexSpace,
    CofaceNeighborhood,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    MemoryNeighborhood,
    NetworkSpace,
    PhaseSpace,
    SimplexSpace,
)
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine._utils import domain_of, symbol
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import BasePass
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile._rewriting_compiler import RewritingCompiler
from skcomplex.core.engine.compile.passes._flatten import FlattenPass
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.engine.problems import ProblemSpec
from skcomplex.core.engine.solvers import GillespieSolver, LammpsSolver, MilsteinSolver, PontryaginSolver, RK4Solver
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
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
from skcomplex.core.specs import (
    AxisSpec,
    IssueSpec,
    MoveSpec,
    ObjectiveSpec,
    ObservableSpec,
    ParameterSpec,
    RoleSpec,
    SamplerSpec,
    StateSpec,
    ValidationReportSpec,
    VisualSpec,
    WienerProcessSpec,
)
from skcomplex.core.types import ElementKind, FormKind, NoiseKind
from skcomplex.systems.biology.common.populations import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import LotkaVolterraSystem, PredationInteraction
from skcomplex.systems.biology.epidemiology import InfectionInteraction, RecoveryInteraction
from skcomplex.systems.economics.finance import AssetElement, GeometricBrownianMotionInteraction
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
from skcomplex.systems.physics.common.particles import (
    ConstantForceInteraction,
    HarmonicPotentialInteraction,
    MassPointElement,
)
from skcomplex.systems.physics.common.quantum import QubitElement
from skcomplex.systems.physics.statistical_mechanics import ParticleSiteElement, ZeroRangeHopInteraction

from ....conftest import CallableGrowthInteraction, HoldingElement, NodeElement, _correlated, _summed, _weighted
from .conftest import PairElement

CUTOFF = 2.5
RECEIVED: dict[str, object] = {}


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


def _growth():
    """Return a system of one growing population."""
    return CompositeSystem(
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
    """Element with one nonnegative state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


class WalkerElement(Element):
    """Element with a vector position."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class CounterElement(Element):
    """Element whose state grows by one every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k

    def _get_laws(self, s):
        """Return the update of the count."""
        return [UpdateLaw(s.k, s.k + 1)]


class GrowthInteraction(Interaction):
    """Exponential growth of the state of its target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s):
        """Return the rate of growth."""
        return [RateLaw(s.target.n, s.r * s.target.n)]


class AntElement(Element):
    """Agent that counts its steps."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        steps: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.steps = steps

    def _get_laws(self, s):
        """Return the rule of the step."""
        return [RuleLaw(self.walk)]

    def walk(self, agent, system):
        """Count one step."""
        agent.steps += 1


def _codes(report):
    """Return the codes of the issues of a report."""
    return [issue.code for issue in report.issues]


class ScaledRateLaw(RateLaw):
    """A rate law of a class the engine has no form for."""


class ScaledStockElement(StockElement):
    """Stock whose decay is a law of a class with no form."""

    def _get_laws(self, s):
        """Return the decay."""
        return [ScaledRateLaw(s.n, -s.n)]


class PlantElement(Element):
    """Element steered by a bounded control input."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(0, 1))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.u = u

    def _get_laws(self, s):
        """Return the rate of the state."""
        return [RateLaw(s.x, s.u - s.x)]


class NoisyPlantElement(PlantElement):
    """Plant whose state is driven by noise."""

    def _get_laws(self, s):
        """Return the laws of the plant with noise."""
        return [*super()._get_laws(s), NoiseLaw(s.x, 0.1)]


class GroupSpinElement(Element):
    """Spin on a group space sampled with the default checkerboard schedule."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.spin = spin

    def _get_laws(self, s):
        """Return the Glauber sampling of a ferromagnetic energy on the generators."""
        memory = MemoryNeighborhood(('a', 'a^-1', 'b', 'b^-1'))
        energy = ObjectiveSpec(-s.spin * sum(neighbors(s.spin, memory, reduce='each')), memory)
        return [SampleLaw(energy, MoveSpec(s.spin), SamplerSpec('glauber', 2.0))]


class MisuseInteraction(Interaction):
    """Rate law that uses the handle of a Wiener process as a value."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        noise: Annotated[WienerProcessSpec, RoleSpec(noise=True)],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.noise = noise

    def _get_laws(self, s):
        """Return the rate."""
        return [RateLaw(s.target.n, s.noise.dW)]


class CoupledInteraction(Interaction):
    """Stratonovich noise on a state, through a Wiener process another component also draws."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        other: Annotated[str, RoleSpec()],
        noise: Annotated[WienerProcessSpec, RoleSpec(noise=True)],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.other = other
        self.noise = noise

    def _get_laws(self, s):
        """Return the noise of the target, whose amplitude reads the other state."""
        return [NoiseLaw(s.target.n, s.other.n, convention='stratonovich', process=s.noise.dW)]


class SingleInteraction(Interaction):
    """Itô noise on one element from a Wiener process."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        noise: Annotated[WienerProcessSpec, RoleSpec(noise=True)],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.noise = noise

    def _get_laws(self, s):
        """Return the noise."""
        return [NoiseLaw(s.target.n, 1.0, process=s.noise.dW)]


class SpreadInteraction(NeighborInteraction):
    """Spread of the state of a node to its neighbours."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.nodes = nodes

    def _get_laws(self, s):
        """Return the spread along one edge."""
        return [RateLaw(s.target.x, s.source.x - s.target.x)]


class ConsensusInteraction(HyperedgeInteraction):
    """Pull of the members of a hyperedge towards their mean."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.nodes = nodes

    def _get_laws(self, s):
        """Return the pull of the members of one hyperedge."""
        return [RateLaw(s.member.x, mean(s.members.x, over=s.members) - s.member.x)]


def _bindings(interaction, space):
    """Return the binding issues of a group of four nodes on a space, coupled by an interaction."""
    nodes = GroupElement('nodes', element=NodeElement, size=4)
    system = CompositeSystem(elements=[nodes], interactions=[interaction], space=space)
    report = PassCompiler().prepare(system).report_
    issues = report.issues
    return [issue for issue in issues if issue.code == 'binding']


class ReadingInteraction(Interaction):
    """Interaction that declares nothing about the states its laws read."""

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return a rate that reads the state `n` of the target."""
        return [RateLaw(s.target.n, -s.target.n)]


class ReadingSourcesInteraction(Interaction):
    """Interaction that reads a state of the elements of a variadic role."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        sources: Annotated[list[str], RoleSpec(variadic=True)],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.sources = sources

    def _get_laws(self, s):
        """Return a rate of the target that reads the state `missing` of its first source."""
        return [RateLaw(s.target.x, s.sources[0].missing)]


class TankElement(Element):
    """Element whose only state is `level`."""

    def __init__(self, label: str, level: Annotated[float, StateSpec()] = 1.0) -> None:
        super().__init__(label)
        self.level = level


def _role_issues(system: CompositeSystem) -> list[IssueSpec]:
    """Return the `role-element` issues of a system."""
    return [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'role-element']


class WalkInteraction(Interaction):
    """Agent rule that acts on the agents of a group, like a movement."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP))],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return a rule that leaves the agents in place."""
        return [RuleLaw(lambda system: None)]


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
    system = CompositeSystem(elements=[QubitElement('q')])
    assert OperatorCompiler().prepare(system).report_.ok
    assert not RewritingCompiler().prepare(system).report_.ok
    compiler = PassCompiler().prepare(system)
    assert not compiler.report_.ok
    with pytest.raises(ValidationError, match=r'compiler=OperatorCompiler\(\)'):
        compiler.run()


def test_compiler_reports_an_observable_of_a_missing_system():
    """Test that an observable labelled with a nested system that does not exist is an issue of the report."""
    observable = ObservableSpec('village/size', lambda s: s.rabbits.x)
    report = PassCompiler(observables=[observable]).prepare(_growth()).report_
    assert [issue.code for issue in report.errors] == ['observable-path']


def _nonlocal_field(label, law, space, **settings):
    """Return a field of one state `u` and one parameter `k` whose state changes at the rate `law(s)`."""

    class NonlocalElement(Element):
        """Nonlocal element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            u: Annotated[object, StateSpec()] = 0.0,
            k: Annotated[object, ParameterSpec()] = -1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.u = u
            self.k = k

        def _get_laws(self, s):
            """Return the law of the nonlocal element."""
            return [RateLaw(s.u, law(s))]

    return FieldElement(label, element=NonlocalElement, space=space, **settings)


def test_compiler_reports_a_negative_screening():
    """Test that a screening whose square is negative, or that reads a state, is a `screening` issue of the law."""
    for screening in (lambda s: sqrt(s.k), lambda s: s.u):
        field = _nonlocal_field(
            'm',
            lambda s, screening=screening: inverse_helmholtz(s.u, screening(s)),
            LatticeSpace(8),
        )
        report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
        assert [(issue.code, issue.path) for issue in report.errors] == [('screening', 'm')]
        assert 'screening' in report.errors[0].message


def test_compiler_reports_a_kernel_that_reads_a_state():
    """Test that a convolution kernel that reads a state is a `kernel-state` issue naming the state."""
    field = _nonlocal_field('m', lambda s: convolve(s.u, s.u * displacement(0)), LatticeSpace(8))
    report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
    assert [(issue.code, issue.path) for issue in report.errors] == [('kernel-state', 'm')]
    assert 'u_m' in report.errors[0].message


def test_compiler_reports_a_two_point_kernel_that_reads_a_state():
    """Test that a two-point kernel that reads a state is a `kernel-state` issue naming the state."""
    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('z', 8, 0.0, 1.0)])
    field = _nonlocal_field('m', lambda s: integral_transform(s.u, s.u * integration_variable('z'), ('z',)), space)
    report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
    assert [(issue.code, issue.path) for issue in report.errors] == [('kernel-state', 'm')]
    assert 'u_m' in report.errors[0].message


def test_compiler_reports_a_two_point_integral_without_class_axes():
    """Test that an integral over class axes of a field on a lattice is a `law-space` issue."""
    field = _nonlocal_field('m', lambda s: integral_transform(s.u, integration_variable('z'), ('z',)), LatticeSpace(8))
    report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
    assert {(issue.code, issue.path) for issue in report.errors} == {('law-space', 'm')}
    assert any('two-point kernel' in issue.message for issue in report.errors)


def test_compiler_reports_a_cumulative_integral_on_a_network():
    """Test that a cumulative integral on a network is a `law-space` issue saying it needs a lattice space."""
    field = _nonlocal_field('net', lambda s: cumulative_integral(s.u, 0), NetworkSpace(nx.path_graph(4)))
    report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
    assert [(issue.code, issue.path) for issue in report.errors] == [('law-space', 'net')]
    assert 'cumulative integral' in report.errors[0].message


def test_compiler_reports_a_local_operator_on_a_simplex():
    """Test that a Laplacian of a field on a simplex is a `law-space` issue asking for finite-volume fluxes."""
    space = SimplexSpace(LatticeSpace((1,)), [AxisSpec('x', 6, 0.0, 1.0), AxisSpec('y', 6, 0.0, 1.0)])
    field = _nonlocal_field('m', lambda s: laplacian(s.u) * coordinate('x'), space)
    report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
    assert [(issue.code, issue.path) for issue in report.errors] == [('law-space', 'm')]
    assert 'finite-volume' in report.errors[0].message


def test_compiler_reports_a_convolution_on_a_network():
    """Test that a convolution on a network is a `law-space` issue saying it needs a lattice space."""
    field = _nonlocal_field('net', lambda s: convolve(s.u, displacement(0)), NetworkSpace(nx.path_graph(4)))
    report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
    assert [(issue.code, issue.path) for issue in report.errors] == [('law-space', 'net')]
    assert 'lattice space' in report.errors[0].message


def test_compiler_reports_spectral_operators_on_edges_that_are_not_periodic():
    """Test that a spectral inversion on a lattice with fixed edges is a `law-space` issue."""
    field = _nonlocal_field(
        'm',
        lambda s: inverse_helmholtz(s.u, 1),
        LatticeSpace(8, boundary='fixed'),
        discretization='spectral',
    )
    report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
    assert [issue.code for issue in report.errors] == ['law-space']


class _CrossInteraction(Interaction):
    """Inversion of the sum of the states of two fields."""

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second

    def _get_laws(self, s):
        """Return the rate of the first field."""
        return [RateLaw(s.first.u, inverse_helmholtz(s.first.u + s.second.u, 1))]


def test_compiler_reports_an_operand_over_two_spaces():
    """Test that an operand reading fields of two spaces is an `operator-spaces` issue."""
    other = _nonlocal_field('b', lambda s: 0 * s.u, LatticeSpace(8))
    field = _nonlocal_field('a', lambda s: 0 * s.u, LatticeSpace(8))
    system = CompositeSystem(elements=[field, other], interactions=[_CrossInteraction('cross', first='a', second='b')])
    report = PassCompiler().prepare(system).report_
    assert [issue.code for issue in report.errors] == ['operator-spaces']


def test_compiler_raises_every_nonlocal_issue_at_once():
    """Test that a convolution on a network and a negative screening raise one error that lists both laws."""
    network = _nonlocal_field('net', lambda s: convolve(s.u, displacement(0)), NetworkSpace(nx.path_graph(4)))
    lattice = _nonlocal_field('grid', lambda s: inverse_helmholtz(s.u, sqrt(s.k)), LatticeSpace(8))
    compiler = PassCompiler().prepare(CompositeSystem(elements=[network, lattice]))
    with pytest.raises(ValidationError, match=r'(?s)net: .*law-space.*grid: .*screening'):
        compiler.run()


class _TensorElement(Element):
    """Element with a symmetric traceless state and a scalar, whose rate a function of the test gives."""

    rate = staticmethod(lambda s: -s.Q)

    def __init__(self, label, Q: Annotated[object, StateSpec(tensor='traceless')] = 0.0):
        super().__init__(label)
        self.Q = Q

    def _get_laws(self, namespace):
        """Return the rate of the tensor state."""
        return [RateLaw(namespace.Q, type(self).rate(namespace), label='relax')]


def _tensor_issues(rate):
    """Return the issues of a field whose tensor state has the given rate."""

    class RatedElement(_TensorElement):
        pass

    RatedElement.rate = staticmethod(rate)
    field = FieldElement('film', element=RatedElement, space=LatticeSpace((4, 4)))
    return PassCompiler().prepare(CompositeSystem(elements=[field])).report_.issues


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

    class ScalarElement(Element):
        def __init__(self, label, u: Annotated[float, StateSpec()] = 0.0):
            super().__init__(label)
            self.u = u

        def _get_laws(self, namespace):
            return [RateLaw(namespace.u, tensor_divergence(namespace.u)[0])]

    field = FieldElement('film', element=ScalarElement, space=LatticeSpace((4, 4)))
    issues = PassCompiler().prepare(CompositeSystem(elements=[field])).report_.issues
    assert [issue.code for issue in issues] == ['law-domain']
    assert 'tensor_divergence' in issues[0].message
    assert 'u_film' in issues[0].message


class _CellSiteElement(Element):
    """Site with a value `u`."""

    def __init__(self, label, visual=None, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.u = u


class _CellLinkElement(Element):
    """Link with a value `a`."""

    def __init__(self, label, visual=None, *, a: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.a = a


class _CellFaceElement(Element):
    """Face with a value `f`."""

    def __init__(self, label, visual=None, *, f: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.f = f


def _cell_groups():
    """Return groups on the sites, the edges, and the hyperedges of the complex of a lattice."""
    space = CellComplexSpace(LatticeSpace((4, 4)))
    return [
        GroupElement('sites', element=_CellSiteElement, space=space, on='sites'),
        GroupElement('links', element=_CellLinkElement, space=space, on='edges'),
        GroupElement('faces', element=_CellFaceElement, space=space, on='hyperedges'),
    ]


def _report_of(system):
    """Return the report of a system."""
    return PassCompiler().prepare(system).report_


def test_report_holds_one_degree_issue_per_law_with_its_own_path_and_fix():
    """Test that a codifferential of a site state and a sampling of a hyperedge group give two issues."""
    interactions = [
        RateInteraction('down', target='sites', state='u', rate=lambda n: codifferential(n.target.u)),
        SamplingInteraction(
            'sample',
            target='faces',
            state='f',
            objective=lambda n: n.target.f**2,
            move='gaussian',
            schedule='colored',
        ),
    ]
    system = CompositeSystem(elements=_cell_groups(), interactions=interactions, time=ContinuousTime())
    issues = [issue for issue in _report_of(system).issues if issue.code == 'law-degree']
    assert sorted(issue.path for issue in issues) == ['down', 'sample']
    assert 'exterior_derivative' in issues[0].fix
    assert 'edges' in issues[1].fix


def test_report_holds_a_degree_issue_for_an_exterior_derivative_of_the_highest_degree():
    """Test that an exterior derivative of a hyperedge state on a complex of dimension 2 is reported."""
    rate = RateInteraction('up', target='faces', state='f', rate=lambda n: exterior_derivative(n.target.f))
    issues = _report_of(CompositeSystem(elements=_cell_groups(), interactions=[rate])).issues
    assert [issue.code for issue in issues if issue.path == 'up'] == ['law-degree']
    assert 'codifferential' in next(issue.fix for issue in issues if issue.path == 'up')


def test_report_holds_a_degree_issue_for_a_value_of_another_degree():
    """Test that a rate of a state from a value of another degree is reported, with or without an operation."""
    twice = RateInteraction(
        'twice',
        target='links',
        state='a',
        rate=lambda n: exterior_derivative(exterior_derivative(n.target.a)),
    )
    bare = RateInteraction('bare', target='sites', state='u', sources=['faces'], rate=lambda n: n.sources[0].f)
    wrong = RateInteraction(
        'wrong',
        target='sites',
        state='u',
        sources=['links'],
        rate=lambda n: hodge_star(n.sources[0].a),
    )
    for interaction in (twice, bare, wrong):
        issues = _report_of(CompositeSystem(elements=_cell_groups(), interactions=[interaction])).issues
        assert [issue.path for issue in issues if issue.code == 'law-degree'] == [interaction.label]


def test_report_holds_no_degree_issue_for_operations_of_the_right_degree():
    """Test that a curl of a curl on links and a gradient of a site state are not reported."""
    interactions = [
        RateInteraction(
            'curl',
            target='links',
            state='a',
            rate=lambda n: codifferential(exterior_derivative(n.target.a)),
        ),
        RateInteraction(
            'gradient',
            target='links',
            state='a',
            sources=['sites'],
            rate=lambda n: exterior_derivative(n.sources[0].u),
        ),
    ]
    issues = _report_of(CompositeSystem(elements=_cell_groups(), interactions=interactions)).issues
    assert not [issue for issue in issues if issue.code == 'law-degree']


def test_report_holds_no_degree_issue_for_a_group_mean_of_another_degree():
    """Test that a mean over the faces of an exterior derivative scales a link rate as a number of no degree."""
    rate = RateInteraction(
        'scaled',
        target='links',
        state='a',
        sources=['faces'],
        rate=lambda n: mean(cos(exterior_derivative(n.target.a)), over=n.sources[0]) * n.target.a,
    )
    issues = _report_of(CompositeSystem(elements=_cell_groups(), interactions=[rate])).issues
    assert not [issue for issue in issues if issue.code == 'law-degree']


def test_report_holds_a_degree_issue_for_a_misapplied_operation_inside_a_group_mean():
    """Test that a codifferential of a site state inside a group mean is still reported."""
    rate = RateInteraction(
        'scaled',
        target='sites',
        state='u',
        rate=lambda n: mean(codifferential(n.target.u)) * n.target.u,
    )
    issues = _report_of(CompositeSystem(elements=_cell_groups(), interactions=[rate])).issues
    assert [issue.path for issue in issues if issue.code == 'law-degree'] == ['scaled']


def test_report_holds_a_space_issue_for_cofaces_outside_a_cell_complex():
    """Test that a coface reduction of a rate or of a sampling outside a cell complex is a law-space issue."""
    space = NetworkSpace(nx.cycle_graph(4))
    sites = GroupElement('sites', element=_CellSiteElement, space=space, on='sites')
    rate = RateInteraction(
        'rate',
        target='sites',
        state='u',
        rate=lambda n: neighbors(n.target.u, over=CofaceNeighborhood()),
    )
    sampling = SamplingInteraction(
        'sample',
        target='sites',
        state='u',
        objective=lambda n: neighbors(n.target.u, over=CofaceNeighborhood()),
        move='gaussian',
        schedule='colored',
    )
    issues = _report_of(CompositeSystem(elements=[sites], interactions=[rate])).issues
    assert [issue.code for issue in issues if issue.path == 'rate'] == ['law-space']
    issues = _report_of(CompositeSystem(elements=[sites], interactions=[sampling], time=DiscreteTime())).issues
    assert [issue.code for issue in issues if issue.path == 'sample'] == ['law-space']


def test_report_holds_a_space_issue_for_the_exterior_calculus_outside_a_cell_complex():
    """Test that an exterior derivative of a state of a plain network is reported as a law-space issue."""
    sites = GroupElement('sites', element=_CellSiteElement, space=NetworkSpace(nx.cycle_graph(4)), on='sites')
    rate = RateInteraction('rate', target='sites', state='u', rate=lambda n: exterior_derivative(n.target.u))
    issues = _report_of(CompositeSystem(elements=[sites], interactions=[rate])).issues
    assert [issue.code for issue in issues if issue.path == 'rate'] == ['law-space']


def test_compiler_reports_a_bad_setting_before_a_missing_system():
    """Test that preparing a compiler without a system names its invalid setting, not the missing system."""
    with pytest.raises(ValidationError, match='backend'):
        PassCompiler(backend='numba').prepare()


class UndeclaredInteraction(Interaction):
    """Interaction that uses a symbol it does not declare."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return a law with an undeclared symbol."""
        return [RateLaw(s.target.x, symbol('k') * s.target.x)]


class DoubleUpdateInteraction(Interaction):
    """Interaction that updates the same state twice."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return two updates of one state."""
        return [UpdateLaw(s.target.price, 1), UpdateLaw(s.target.price, 2)]


def _ecology(growth):
    """Return a system of one population under a growth interaction."""
    return CompositeSystem(elements=[PopulationElement('rabbits', x=10)], interactions=[growth])


class HubElement(Element):
    """Hub element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y


class SiteDriveInteraction(Interaction):
    """Growth of a hub at the rate of the state of one site."""

    def __init__(
        self,
        label: str,
        *,
        hub: Annotated[str, RoleSpec()],
        site: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.hub = hub
        self.site = site

    def _get_laws(self, s):
        """Return the growth of the hub."""
        return [RateLaw(s.hub.y, s.site.x)]


class SitesDriveInteraction(Interaction):
    """Growth of a hub at the rate of the summed states of several sites."""

    def __init__(
        self,
        label: str,
        *,
        hub: Annotated[str, RoleSpec()],
        sites: Annotated[list[str], RoleSpec(variadic=True)],
    ) -> None:
        super().__init__(label)
        self.hub = hub
        self.sites = sites

    def _get_laws(self, s):
        """Return the growth of the hub."""
        return [RateLaw(s.hub.y, sum((item.x for item in s.sites), 0))]


def _hub_and_sites(interaction):
    """Return a hub and a group of five sites with states 0 to 4, coupled by an interaction."""
    return CompositeSystem(
        elements=[
            GroupElement('sites', element=NodeElement, size=5, values={'x': np.arange(5.0)}),
            HubElement('hub'),
        ],
        interactions=[interaction],
    )


def _compile_time(n_links):
    """Return the shortest of two compile times of a zero-range ring written with one interaction per link."""
    sites = GroupElement(
        'sites',
        element=ParticleSiteElement,
        space=LatticeSpace(n_links),
        values={'x': np.ones(n_links)},
        on='sites',
    )
    links = [
        ZeroRangeHopInteraction(f'hop_{index}', link=[f'sites[{index}]', f'sites[{(index + 1) % n_links}]'])
        for index in range(n_links)
    ]
    system = CompositeSystem(elements=[sites], interactions=links)
    durations = []
    for _ in range(2):
        start = time.perf_counter()
        PassCompiler(backend=NumpyBackend()).prepare(system).run()
        durations.append(time.perf_counter() - start)
    return min(durations)


class LennardJonesInteraction(PairInteraction):
    """Lennard-Jones pair potential."""

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            potential=self._get_pair_energy,
            period=period,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    def _get_pair_energy(self, s, r):
        """Return `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`."""
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


class DoublingInteraction(Interaction):
    """Interaction with a callable rate over a group."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return a callable rate equal to twice the member counts."""
        name = f'{self.target}.x'
        return [RateLaw(s.target.x, lambda t, values: 2 * values[name])]


class BirthInteraction(Interaction):
    """Stochastic birth of every member of a group."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _get_laws(self, s):
        """Return one jump per member."""
        return [JumpLaw(s.rate * s.target.x, {s.target.x: 1})]


class VolatilityInteraction(Interaction):
    """Noise on every member of a group."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the noise amplitude `x`."""
        return [NoiseLaw(s.target.x, s.target.x)]


class HalvingInteraction(Interaction):
    """Discrete halving of every member of a group."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the next value `n / 2`."""
        return [UpdateLaw(s.target.x, s.target.x / 2)]


class SingleUpdateFromGroupInteraction(Interaction):
    """Update of a single element from a group, which has no meaning."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        total: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.total = total

    def _get_laws(self, s):
        """Return an update of the single element."""
        return [UpdateLaw(s.total.x, s.target.x)]


def _lennard_jones_gas(n, seed=0):
    """Return a gas of Lennard-Jones particles on a jittered lattice."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n] * 1.2
    return CompositeSystem(
        elements=[
            GroupElement(
                'gas',
                element=MassPointElement,
                size=n,
                values={'x': grid + rng.normal(0, 0.05, (n, 2)), 'x_d1': rng.normal(0, 0.5, (n, 2))},
            ),
        ],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )


class RecorderInteraction(Interaction):
    """Interaction that records the symbols it receives and adds no change."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Record the symbols and return a zero rate on the velocity."""
        RECEIVED.update(x=s.target.x, m=s.target.m)
        return [RateLaw(s.target.x, 0)]


@pytest.mark.parametrize(
    ('system', 'form'),
    [
        (_ecology(LogisticGrowthInteraction('growth', target='rabbits')), FormKind.ODE),
        (
            CompositeSystem(
                elements=[AssetElement('stock', price=100)],
                interactions=[GeometricBrownianMotionInteraction('gbm', target='stock')],
            ),
            FormKind.SDE,
        ),
        (
            CompositeSystem(
                elements=[AmountElement('I', x=5), AmountElement('R')],
                interactions=[RecoveryInteraction('rec', infected='I', recovered='R', stochastic=True)],
            ),
            FormKind.JUMP,
        ),
        (
            CompositeSystem(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')]),
            FormKind.DISCRETE,
        ),
        (CompositeSystem(elements=[PopulationElement('a')]), FormKind.ODE),
    ],
)
def test_compiler_infers_form(system, form):
    """Test that the form follows from the law kinds."""
    assert PassCompiler().prepare(system).run().problem_.form == form


def test_compiler_mixed_kinds_raises_error():
    """Test that updates and rates in one system are rejected."""
    system = CompositeSystem(
        elements=[MarketElement('m'), PopulationElement('a')],
        interactions=[CobwebInteraction('c', target='m'), ExponentialGrowthInteraction('g', target='a')],
    )
    with pytest.raises(IncompatibleLawsError, match='RateLaw: a; UpdateLaw: m'):
        PassCompiler().prepare(system).run()


def test_compiler_names_nested_states_by_path():
    """Test that nested states are named by their path."""
    system = CompositeSystem(systems=[CompositeSystem('meadow', elements=[PopulationElement('rabbits', x=3)])])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.states.names == ['meadow/rabbits.x']
    assert problem.initial_state.tolist() == [3.0]


def test_compiler_couples_subsystems():
    """Test that an interaction couples elements of two subsystems."""
    system = CompositeSystem(
        systems=[
            CompositeSystem('forest', elements=[PopulationElement('foxes', x=2)]),
            CompositeSystem('meadow', elements=[PopulationElement('rabbits', x=5)]),
        ],
        interactions=[PredationInteraction('hunt', predator='forest/foxes', prey='meadow/rabbits', a=1.0, b=0.5)],
    )
    problem = PassCompiler().prepare(system).run().problem_
    derivative = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert derivative.tolist() == [5.0, -10.0]


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
    system = CompositeSystem(elements=[HoldingElement('rabbits', n=10.0)], interactions=[growth])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [30.0]
    assert problem.has_callables
    assert problem.equations == {}


def test_compiler_rates_add_up():
    """Test that two forces on one particle add up."""
    system = CompositeSystem(
        elements=[MassPointElement('p', x=1.0, m=2.0)],
        interactions=[
            ConstantForceInteraction('push', target='p', F=4.0),
            HarmonicPotentialInteraction('trap', target='p', k=6.0),
        ],
        space=ContinuousSpace(n_dim=1),
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [0.0, -1.0]


def test_compiler_keeps_oscillator_equations():
    """Test that the velocity equation of an oscillator is `-k * (x - center) / m`."""
    system = CompositeSystem(
        elements=[MassPointElement('p')],
        interactions=[HarmonicPotentialInteraction('trap', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    problem = PassCompiler().prepare(system).run().problem_
    x, v = problem.state_symbols
    k = problem.symbols['trap.k']
    m = problem.symbols['p.m']
    (center,) = problem.symbols['trap.center']
    assert problem.equations['p.x[0]'] == v
    assert (problem.equations['p.x_d1[0]'] + k * (x - center) / m).simplify_full() == 0


def test_compiler_vector_state_takes_space_dimension():
    """Test that vector states have one component per dimension."""
    system = CompositeSystem(elements=[MassPointElement('p')], space=ContinuousSpace(n_dim=2))
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.states.component_names == ['p.x[0]', 'p.x[1]', 'p.x_d1[0]', 'p.x_d1[1]']


def test_compiler_vector_state_without_space_raises_error():
    """Test that a vector state needs a space."""
    with pytest.raises(SpaceError, match='must have a space'):
        PassCompiler().prepare(CompositeSystem(elements=[MassPointElement('p')])).run()


@pytest.mark.parametrize('m', [-2.0, 0.0, 'm'])
def test_compiler_value_outside_domain_raises_error(m):
    """Test that a parameter outside its domain is rejected."""
    system = CompositeSystem(elements=[MassPointElement('p', m=m)], space=ContinuousSpace(n_dim=1))
    with pytest.raises(ValidationError, match=r'p\.m'):
        PassCompiler().prepare(system).run()


def test_compiler_wrong_shape_raises_error():
    """Test that a vector value of the wrong size is rejected."""
    system = CompositeSystem(elements=[MassPointElement('p', x=[1, 2, 3])], space=ContinuousSpace(n_dim=2))
    with pytest.raises(ValidationError, match='shape'):
        PassCompiler().prepare(system).run()


def test_compiler_unknown_role_raises_error():
    """Test that a role must name an element."""
    system = _ecology(LogisticGrowthInteraction('growth', target='foxes'))
    with pytest.raises(RoleError, match="'target' to 'foxes'"):
        PassCompiler().prepare(system).run()


def test_compiler_duplicate_label_raises_error():
    """Test that siblings must have distinct labels."""
    system = CompositeSystem(elements=[PopulationElement('a'), PopulationElement('a')])
    with pytest.raises(LabelError, match="'a'"):
        PassCompiler().prepare(system).run()


def test_compiler_invalid_label_raises_error():
    """Test that labels must be words."""
    with pytest.raises(LabelError, match='not valid'):
        PassCompiler().prepare(CompositeSystem(elements=[PopulationElement('a.b')])).run()


def test_compiler_undeclared_symbol_raises_error():
    """Test that laws may only use declared symbols."""
    with pytest.raises(ValidationError, match='symbols k'):
        PassCompiler().prepare(_ecology(UndeclaredInteraction('bad', target='rabbits'))).run()


def test_compiler_double_update_raises_error():
    """Test that a state takes one update law."""
    system = CompositeSystem(elements=[MarketElement('m')], interactions=[DoubleUpdateInteraction('bad', target='m')])
    with pytest.raises(ValidationError, match='second time'):
        PassCompiler().prepare(system).run()


def test_compiler_jump_stoichiometry():
    """Test that jump changes become rows of the stoichiometry matrix."""
    system = CompositeSystem(
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
    system = _hub_and_sites(SiteDriveInteraction('link', hub='hub', site='sites[3]'))
    plain = PassCompiler().prepare(system).run().problem_
    system = _hub_and_sites(SitesDriveInteraction('link', hub='hub', sites=['sites[3]']))
    variadic = PassCompiler().prepare(system).run().problem_
    for problem in (plain, variadic):
        drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
        assert problem.states.read(drift, 'hub.y') == 3.0
    assert plain.interactions[0].roles['site'] == 'sites[3]'


def test_compile_plain_role_past_group_end_raises_error():
    """Test that a plain role bound past the end of a group names the interaction, role, reference, and size."""
    with pytest.raises(RoleError, match=r"link binds 'sites\[7\]' to its role 'site', but sites holds 5 members"):
        PassCompiler().prepare(_hub_and_sites(SiteDriveInteraction('link', hub='hub', site='sites[7]'))).run()


def test_compile_per_entry_parameter_with_wrong_length_raises_error():
    """Test that a per-entry parameter of another length than its role names the interaction and both lengths."""
    with pytest.raises(ValidationError, match=r"sum .*'weights' 2 values.*'inputs' binds 3 entries"):
        PassCompiler().prepare(_summed(['a', 'b', 'c'], weights=[1.0, 2.0])).run()


def test_compile_per_entry_parameter_has_one_component_per_entry():
    """Test that a per-entry parameter is laid out with one addressable component per entry."""
    problem = PassCompiler().prepare(_summed(['a', 'b', 'c'], weights=[1.0, 2.0, 3.0])).run().problem_
    assert problem.parameters.component_names == ['sum.weights[0]', 'sum.weights[1]', 'sum.weights[2]']


@pytest.mark.benchmark
def test_compiler_time_grows_linearly_with_interactions():
    """Test that compiling 800 one-link interactions takes at most 2.5 times as long as compiling 400."""
    assert _compile_time(800) <= 2.5 * _compile_time(400)


@pytest.mark.benchmark
def test_compile_group_time_does_not_grow_with_size():
    """Test that compiling 100000 members takes no longer than 10 members within a factor of 2."""

    def compile_time(n):
        start = time.perf_counter()
        system = CompositeSystem(
            elements=[GroupElement('gas', element=MassPointElement, size=n)],
            space=ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting'),
        )
        PassCompiler().prepare(system).run()
        return time.perf_counter() - start

    compile_time(10)
    assert compile_time(100_000) <= 2 * compile_time(10) + 0.05


def test_compile_group_names_states_by_group_path():
    """Test that a group state is named by its group path with the group shape."""
    system = CompositeSystem(
        elements=[GroupElement('gas', element=MassPointElement, size=500)],
        space=ContinuousSpace(n_dim=2),
    )
    problem = PassCompiler().prepare(system).run().problem_
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
    system = CompositeSystem(
        elements=[
            GroupElement('a', element=MassPointElement, size=1, values={'x': [[0.0, 0.0]]}),
            GroupElement('b', element=MassPointElement, size=1, values={'x': [[1.5, 0.0]]}),
        ],
        interactions=[LennardJonesInteraction('lj', first='a', second='b', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )
    problem = PassCompiler().prepare(system).run().problem_
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    a, b = problem.states.read(drift, 'a.x_d1'), problem.states.read(drift, 'b.x_d1')
    assert a[0, 0] > 0
    assert np.allclose(a, -b)


def test_compile_pair_forces_use_minimum_image():
    """Test that periodic pair interactions use minimum image distances."""
    system = CompositeSystem(
        elements=[GroupElement('gas', element=MassPointElement, size=2, values={'x': [[0.1, 0.5], [9.6, 0.5]]})],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF, period=10.0)],
        space=ContinuousSpace(n_dim=2),
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
    system = CompositeSystem(
        elements=[GroupElement('gas', element=MassPointElement, size=4)],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting'),
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.form.name == 'ODE'
    assert problem.boundary is not None


@pytest.mark.parametrize(('role', 'wrapped'), [('position', 0.25), ('site', 0.25), (None, 1.25)])
def test_compile_periodic_box_wraps_places_by_their_role(role, wrapped):
    """Test that a periodic box wraps a vector state with the role position or site, and no other."""

    class PlacedElement(Element):
        """Element with a vector state of the role under test."""

        def __init__(self, label, visual=None, *, x: Annotated[object, StateSpec(vector=True, role=role)] = 0.0):
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the rest of the state."""
            return [RateLaw(s.x, 0.0 * s.x)]

    system = CompositeSystem(
        elements=[GroupElement('gas', element=PlacedElement, size=1, values={'x': [[1.25, 0.5]]})],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
    )
    problem = PassCompiler().prepare(system).run().problem_
    state = problem.initial_state
    if problem.boundary is not None:
        state = problem.boundary(0.0, state, problem.initial_parameters)
    assert problem.states.read(state, 'gas.x')[0, 0] == pytest.approx(wrapped)


def test_compile_periodic_box_wraps_positions():
    """Test that a periodic box wraps a position that left the box."""
    system = CompositeSystem(
        elements=[
            GroupElement(
                'gas',
                element=MassPointElement,
                size=1,
                values={'x': [[1.25, 0.5]], 'x_d1': [[1.0, 0.0]]},
            ),
        ],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
    )
    problem = PassCompiler().prepare(system).run().problem_
    state = problem.boundary(0.0, problem.initial_state, problem.initial_parameters)
    assert np.allclose(problem.states.read(state, 'gas.x'), [[0.25, 0.5]])
    assert np.allclose(problem.states.read(state, 'gas.x_d1'), [[1.0, 0.0]])


def test_compile_group_callable_rate():
    """Test that a callable law over a group returns one value per member."""
    system = CompositeSystem(
        elements=[GroupElement('herds', element=PopulationElement, size=3, values={'x': [1.0, 2.0, 3.0]})],
        interactions=[DoublingInteraction('d', target='herds')],
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [2.0, 4.0, 6.0]
    assert problem.has_callables


def test_compile_group_jumps_have_one_channel_per_member():
    """Test that a jump law over a group has one channel per member."""
    system = CompositeSystem(
        elements=[GroupElement('herds', element=PopulationElement, size=3, values={'x': [1.0, 2.0, 3.0]})],
        interactions=[BirthInteraction('birth', target='herds')],
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.propensities(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, 3.0]
    assert problem.stoichiometry.tolist() == np.eye(3).tolist()


def test_compile_group_noise_has_one_process_per_member():
    """Test that a noise law over a group drives every member with its own process."""
    system = CompositeSystem(
        elements=[GroupElement('herds', element=PopulationElement, size=3, values={'x': [1.0, 2.0, 3.0]})],
        interactions=[VolatilityInteraction('vol', target='herds')],
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.noise_rows.tolist() == [0, 1, 2]
    assert problem.diffusion(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, 3.0]


def test_compile_group_updates_every_member():
    """Test that a discrete update over a group updates every member."""
    system = CompositeSystem(
        elements=[GroupElement('herds', element=PopulationElement, size=2, values={'x': [4.0, 8.0]})],
        interactions=[HalvingInteraction('half', target='herds')],
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.update(0.0, problem.initial_state, problem.initial_parameters).tolist() == [2.0, 4.0]
    assert problem.equations['herds.x[i]'] == problem.symbols['herds.x'] / 2


def test_compile_update_of_single_element_from_group_raises_error():
    """Test that an update cannot set a single element from a group."""
    system = CompositeSystem(
        elements=[GroupElement('herds', element=PopulationElement, size=2), PopulationElement('total')],
        interactions=[SingleUpdateFromGroupInteraction('sum', target='herds', total='total')],
    )
    with pytest.raises(ValidationError, match='single element from a group'):
        PassCompiler().prepare(system).run()


def test_compile_boundary_twice_raises_error():
    """Test that a state takes one boundary law."""

    class WrapInteraction(Interaction):
        """Wrap interaction."""

        def __init__(
            self,
            label: str,
            *,
            target: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.target = target

        def _get_laws(self, s):
            """Return the laws of the wrap interaction."""
            return [BoundaryLaw(s.target.x[0], mod(s.target.x[0], 1))]

    system = CompositeSystem(
        elements=[GroupElement('gas', element=MassPointElement, size=2)],
        interactions=[WrapInteraction('wrap', target='gas')],
        space=ContinuousSpace(extent=(1.0,), boundary='reflecting'),
    )
    with pytest.raises(ValidationError, match='second time'):
        PassCompiler().prepare(system).run()


def test_compiler_without_processes_keeps_identity_columns():
    """Test that noise laws that name no process get one process per amplitude and no correlated block."""

    class WalkerElement(Element):
        """Walker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the walker element."""
            return [NoiseLaw(s.x, 1.0), NoiseLaw(s.x, 2.0)]

    system = CompositeSystem(elements=[GroupElement('walkers', element=WalkerElement, size=3)])
    problem = PassCompiler().prepare(system).run().problem_
    np.testing.assert_array_equal(problem.noise_columns, np.arange(6))
    assert problem.n_processes == 6
    assert problem.noise_blocks == ()


def test_compiler_shares_local_process():
    """Test that two laws of one element on one local process share one column."""
    problem = PassCompiler().prepare(CompositeSystem(elements=[PairElement('pair')])).run().problem_
    np.testing.assert_array_equal(problem.noise_columns, [0, 0])
    assert problem.n_processes == 1


def test_compiler_gives_members_their_own_local_process():
    """Test that a local process of a group law gives every member its own process."""
    system = CompositeSystem(elements=[GroupElement('pairs', element=PairElement, size=3)])
    problem = PassCompiler().prepare(system).run().problem_
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
    system = CompositeSystem(elements=[PairElement('pair', conventions=('stratonovich', 'stratonovich'))])
    problem = PassCompiler().prepare(system).run().problem_
    drift = problem.drift(0.0, np.asarray([2.0, 3.0]), problem.initial_parameters)
    np.testing.assert_allclose(drift, [0.5 * 0.5 * 0.5 * 2.0, 0.5 * 0.5 * 0.5 * 3.0])


def test_stratonovich_correction_of_several_laws_adds_up():
    """Test that two Stratonovich laws with independent processes on one state compile and add their corrections."""

    class TwiceElement(Element):
        """Twice element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the twice element."""
            return [
                NoiseLaw(s.x, s.x, convention='stratonovich'),
                NoiseLaw(s.x, 2 * s.x, convention='stratonovich'),
            ]

    problem = PassCompiler().prepare(CompositeSystem(elements=[TwiceElement('twice')])).run().problem_
    drift = problem.drift(0.0, np.asarray([2.0]), problem.initial_parameters)
    assert drift[0] == pytest.approx(0.5 * 2.0 + 0.5 * 4 * 2.0)


def test_stratonovich_callable_amplitude_raises_an_error_naming_the_drift_correction():
    """Test that a Stratonovich noise law with a callable amplitude raises an error instead of running as Itô."""

    class CallableNoiseElement(Element):
        """Element whose noise amplitude is a callable of time and values."""

        def __init__(self, label: str, *, x: Annotated[object, StateSpec()] = 1.0) -> None:
            super().__init__(label)
            self.x = x

        def _get_laws(self, s):
            """Return a Stratonovich noise law with a callable amplitude."""
            return [NoiseLaw(s.x, lambda t, values: values['noisy.x'], convention='stratonovich')]

    with pytest.raises(ValidationError, match='drift correction'):
        PassCompiler().prepare(CompositeSystem(elements=[CallableNoiseElement('noisy')])).run()


def test_components_receive_sage_symbols_and_vectors():
    """Test that scalar symbols are Sage expressions with their domains and vector states are Sage vectors."""
    system = CompositeSystem(
        elements=[MassPointElement('p')],
        interactions=[RecorderInteraction('record', target='p')],
        space=ContinuousSpace(n_dim=3),
    )
    PassCompiler().prepare(system).run()
    assert isinstance(RECEIVED['x'], FreeModuleElement)
    assert len(RECEIVED['x']) == 3
    assert all(isinstance(component, Expression) and component.is_symbol() for component in RECEIVED['x'])
    assert RECEIVED['m'].is_real()
    assert domain_of(RECEIVED['m']) is RR


@pytest.mark.benchmark
def test_small_system_drift_is_fast():
    """Test that the NumPy drift of 20 logistic populations takes at most 250 microseconds and is correct."""
    system = CompositeSystem(
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
    compilation = compiler._get_new_compilation(CompositeSystem())
    compiler._run_passes(compilation, until='lowered')
    assert log == ['Layout', 'Lower']
    compiler._run_passes(compilation)
    assert log == ['Layout', 'Lower', 'Problem']


def test_compiler_flattens_the_system_once(monkeypatch):
    """Test that validating and compiling a system flattens its tree once."""
    calls = []
    flatten = FlattenPass._flatten_system

    def counted(*args, **kwargs):
        calls.append(args[0])
        return flatten(*args, **kwargs)

    monkeypatch.setattr(FlattenPass, '_flatten_system', staticmethod(counted))
    PassCompiler(backend=NumpyBackend()).prepare(LotkaVolterraSystem().prepare().system_).run()
    assert len(calls) == 1


def test_valid_system_has_no_issues():
    """Test that a system of stocks with growth validates without issues."""
    system = CompositeSystem(
        elements=[StockElement('rabbits')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    report = PassCompiler().prepare(system).report_
    assert report.ok
    assert report.issues == ()


def test_several_errors_are_reported_at_once():
    """Test that an unbound role and a vector state without a space are both reported."""
    system = CompositeSystem(
        elements=[StockElement('rabbits'), WalkerElement('walker')],
        interactions=[GrowthInteraction('growth', target='foxes')],
    )
    report = PassCompiler().prepare(system).report_
    assert sorted(_codes(report)) == ['role', 'space-missing']
    assert {issue.path for issue in report.errors} == {'growth', 'walker'}
    assert all(issue.fix for issue in report.issues)


def test_unsupported_solver_lists_capable_solvers():
    """Test that a variable group validated with RK4 reports the unsupported components and capable solvers."""
    system = CompositeSystem(elements=[GroupElement('ants', element=AntElement, size=3, variable=True)])
    report = PassCompiler(solver=RK4Solver()).prepare(system).report_
    issue = next(issue for issue in report.errors if 'laws' in issue.message)
    assert issue.code == 'solver-unsupported'
    assert 'RuleLaw' in issue.message
    assert 'RuleLaw' in issue.fix


def test_mixed_laws_suggest_partitions():
    """Test that rates and updates in one system without children are reported with both components."""
    system = CompositeSystem(
        elements=[StockElement('rabbits'), CounterElement('clock')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    (issue,) = PassCompiler().prepare(system).report_.errors
    assert issue.code == 'laws-mixed'
    assert 'RateLaw: rabbits' in issue.message
    assert 'UpdateLaw: clock' in issue.message
    assert 'nested system' in issue.fix


def test_law_of_a_class_with_no_form_names_its_class():
    """Test that a law of a class with no form raises the unregistered law error, not a mix of laws."""
    system = CompositeSystem(elements=[ScaledStockElement('rabbits'), CounterElement('clock')])
    (issue,) = PassCompiler().prepare(system).report_.errors
    assert (issue.code, issue.error) == ('law-unregistered', UnregisteredLawError)
    with pytest.raises(UnregisteredLawError, match=r'rabbits: The law class ScaledRateLaw has no form'):
        PassCompiler().prepare(system).run()


def test_mixed_laws_in_nested_systems_are_valid():
    """Test that rates and updates in sibling nested systems validate as coupled partitions."""
    system = CompositeSystem(
        systems=[
            CompositeSystem(
                'meadow',
                elements=[StockElement('rabbits')],
                interactions=[GrowthInteraction('growth', target='rabbits')],
            ),
            CompositeSystem('clock', elements=[CounterElement('counter')]),
        ],
    )
    assert PassCompiler().prepare(system).report_.ok


def test_validate_control_outside_bounds():
    """Test that an adaptive parameter starting outside its bounds is one error at the parameter, naming its bounds."""
    (issue,) = PassCompiler().prepare(CompositeSystem(elements=[PlantElement('plant', u=2.0)])).report_.errors
    assert issue.path == 'plant.u'
    assert issue.code == 'parameter-bounds'
    # typing caches equal annotations, so the bounds print as those of the first equal spec: (0, 1) or (0.0, 1.0)
    assert re.search(r'\(0(\.0)?, 1(\.0)?\)', issue.fix)


def test_validate_controls_with_an_optimal_control_solver():
    """Test that a system with bounded controls is valid for an optimal control solver and for a stepper."""
    solver = PontryaginSolver(objective=lambda s: s.plant.x**2)
    assert PassCompiler(solver=solver).prepare(CompositeSystem(elements=[PlantElement('plant')])).report_.ok
    assert PassCompiler(solver=RK4Solver()).prepare(CompositeSystem(elements=[PlantElement('plant')])).report_.ok


def test_validate_noise_with_an_optimal_control_solver():
    """Test that a system with a noise law validated with Pontryagin reports that the solver lacks noise laws."""
    solver = PontryaginSolver(objective=lambda s: s.plant.x**2)
    issues = PassCompiler(solver=solver).prepare(CompositeSystem(elements=[NoisyPlantElement('plant')])).report_.errors
    issue = next(issue for issue in issues if issue.code == 'solver-unsupported')
    assert 'PontryaginSolver does not support the laws NoiseLaw' in issue.message


def test_group_space_reports_the_checkerboard_schedule():
    """Test that a sample with the checkerboard schedule on a group space is one issue naming the other schedules."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2)
    system = CompositeSystem(elements=[GroupElement('spins', element=GroupSpinElement, space=space, on='sites')])
    report = PassCompiler().prepare(system).report_
    issues = report.issues
    (issue,) = [issue for issue in issues if issue.code == 'schedule-unsupported']
    assert issue.path == 'spins'
    assert "'synchronous'" in issue.fix
    assert "'random-sequential'" in issue.fix


def test_group_space_reports_an_image_visual():
    """Test that an image visual on a group space is reported with the Cayley visual as its fix."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2)
    group = GroupElement('spins', element=GroupSpinElement, space=space, visual=VisualSpec('image'), on='sites')
    system = CompositeSystem(elements=[group])
    report = PassCompiler().prepare(system).report_
    issue = next(issue for issue in report.issues if issue.code == 'visual-unsupported')
    assert "'cayley'" in issue.fix


def test_group_space_reports_a_ball_above_the_site_limit():
    """Test that a ball of the free group of rank 3 with more sites than the limit is reported."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b', 'c']), ball=12, max_sites=100)
    issues = PassCompiler().prepare(CompositeSystem(space=space)).report_.issues
    assert [issue.code for issue in issues] == ['space-too-large']
    assert 'more than 100 sites' in issues[0].message


def test_lattice_with_a_bad_boundary_is_reported():
    """Test that a lattice whose boundary is not supported is a space-invalid error that lists the boundaries."""
    space = LatticeSpace((4, 4), boundary='spherical')
    (issue,) = PassCompiler().prepare(CompositeSystem(elements=[StockElement('rabbits')], space=space)).report_.errors
    assert issue.code == 'space-invalid'
    assert "'spherical'" in issue.message
    assert 'periodic, fixed, reflecting' in issue.message


def test_invalid_space_among_other_errors_is_reported():
    """Test that an invalid lattice and an unbound role are both reported, and compiling raises the validation error."""
    system = CompositeSystem(
        elements=[StockElement('rabbits')],
        interactions=[GrowthInteraction('growth', target='hares')],
        space=LatticeSpace((4, 4), boundary='spherical'),
    )
    assert {issue.code for issue in PassCompiler().prepare(system).report_.errors} == {'space-invalid', 'role'}
    with pytest.raises(ValidationError, match='space-invalid'):
        PassCompiler().prepare(system).run()


def test_validation_keeps_the_planned_compilation():
    """Test that validation returns the compilation state planned once, which holds the nodes and the plan."""
    system = CompositeSystem(
        elements=[StockElement('rabbits')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    compilation = PassCompiler().prepare(system).compilation_
    assert [node.path for node in compilation.nodes] == ['rabbits', 'growth']
    assert [partition.name for partition in compilation.plan.partitions] == [system.label]


def test_validate_reports_mixed_conventions():
    """Test that laws of different conventions on one process are reported with both laws and the process."""
    system = CompositeSystem(elements=[PairElement('pair', conventions=('ito', 'stratonovich'))])
    report = PassCompiler().prepare(system).report_
    issues = report.issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-convention-mixed']
    assert 'ito NoiseLaw of pair' in issue.message
    assert 'stratonovich NoiseLaw of pair' in issue.message
    assert "'common'" in issue.message
    assert 'one convention' in issue.fix


def test_validate_reports_correction_out_of_scope():
    """Test that a Stratonovich law reading a state another component drives on its process is reported."""
    system = CompositeSystem(
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
    system = _correlated(covariance)
    report = PassCompiler().prepare(system).report_
    assert [issue.code for issue in report.issues] == ['wiener-covariance']


def test_validate_reports_handle_as_value():
    """Test that a Wiener handle used inside a rate law is reported with the law."""
    system = CompositeSystem(
        elements=[HoldingElement('a')],
        interactions=[MisuseInteraction('misuse', target='a', noise=WienerProcessSpec('market'))],
    )
    report = PassCompiler().prepare(system).report_
    (issue,) = [issue for issue in report.issues if issue.code == 'wiener-handle']
    assert 'RateLaw of misuse' in issue.message
    assert 'only the process of a noise law' in issue.message


def test_validate_reports_wiener_under_discrete_time():
    """Test that a `WienerProcessSpec` in a system of discrete time is reported with the fix of continuous time."""

    class CounterElement(Element):
        """Counter element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            n: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.n = n

        def _get_laws(self, s):
            """Return the laws of the counter element."""
            return [UpdateLaw(s.n, 1)]

    system = CompositeSystem(
        elements=[CounterElement('counter')],
        interactions=[SingleInteraction('single', target='counter', noise=WienerProcessSpec('market'))],
        time=DiscreteTime(),
    )
    (issue,) = [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'wiener-time']
    assert issue.path == 'market'
    assert 'continuous time' in issue.fix


def test_validate_reports_milstein_with_shared_process():
    """Test that Milstein with a shared process gives one error that names the law and suggests other solvers."""
    system = CompositeSystem(elements=[PairElement('pair')])
    report = PassCompiler(solver=MilsteinSolver()).prepare(system).report_
    (issue,) = report.issues
    assert issue.code == 'noise-unsupported'
    assert 'NoiseLaw of pair' in issue.message
    assert 'MilsteinSolver' in issue.message
    assert 'shared noise' in issue.fix


def test_validate_reports_lammps_with_shared_process():
    """Test that the LAMMPS backend with a shared process is reported with the backends that support it."""
    issues = PassCompiler(solver=LammpsSolver()).prepare(CompositeSystem(elements=[PairElement('pair')])).report_.issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-unsupported']
    assert 'NoiseLaw of pair' in issue.message
    assert 'LammpsSolver' in issue.message


def test_validate_accepts_correlated_noise_with_automatic_solver():
    """Test that the automatic SDE solver declares correlated noise, so validation accepts it."""
    assert PassCompiler().prepare(_correlated([[1.0, 0.5], [0.5, 1.0]])).report_.ok
    assert NoiseKind.CORRELATED in DEFAULT_SOLVERS[FormKind.SDE]._get_support().noise


def test_observable_among_the_laws_is_rejected():
    """Test that an observable returned from `laws` is an error that names the component and the laws it takes."""

    class MixedElement(Element):
        """Mixed element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            a: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.a = a

        def _get_laws(self, s):
            """Return the laws of the mixed element."""
            return [RateLaw(s.a, -s.a), ObservableSpec('level', s.a)]

    (issue,) = PassCompiler().prepare(CompositeSystem(elements=[MixedElement('tank')])).report_.errors
    assert issue.path == 'tank'
    assert '`_get_laws` of tank must return a list of' in issue.message
    with pytest.raises(ValidationError, match='tank'):
        PassCompiler().prepare(CompositeSystem(elements=[MixedElement('tank')])).run()


def test_neighbor_interaction_without_neighbours_takes_no_laws_per_edge():
    """Test that an interaction over neighbours of members on no lattice or network binds but has no laws per edge."""
    assert _bindings(SpreadInteraction('spread', nodes='nodes'), None) == []
    nodes = GroupElement('nodes', element=NodeElement, size=4)
    system = CompositeSystem(elements=[nodes], interactions=[SpreadInteraction('spread', nodes='nodes')])
    with pytest.raises(ValidationError, match='laws per edge'):
        PassCompiler().prepare(system).run()


def test_neighbor_interaction_on_a_missing_layer_is_reported():
    """Test that an interaction over neighbours on a layer the network does not have names the layers it has."""
    (issue,) = _bindings(SpreadInteraction('spread', nodes='nodes', layer='work'), NetworkSpace(nx.cycle_graph(4)))
    assert "the layer 'work'" in issue.message
    assert 'LayerSpec' in issue.fix


def test_neighbor_interaction_on_a_network_is_valid():
    """Test that an interaction over neighbours of members on a network has no binding issue."""
    assert _bindings(SpreadInteraction('spread', nodes='nodes'), NetworkSpace(nx.cycle_graph(4))) == []


def test_hyperedge_interaction_without_hyperedges_is_reported():
    """Test that an interaction over hyperedges of members on a network without hyperedges is reported."""
    (issue,) = _bindings(ConsensusInteraction('consensus', nodes='nodes'), NetworkSpace(nx.cycle_graph(4)))
    assert 'a network space with hyperedges' in issue.message


def test_hyperedge_interaction_on_hyperedges_is_valid():
    """Test that an interaction over hyperedges of members on a network with hyperedges has no binding issue."""
    space = NetworkSpace(nx.cycle_graph(4), hyperedges=[(0, 1, 2), (1, 3)])
    assert _bindings(ConsensusInteraction('consensus', nodes='nodes'), space) == []


def test_validate_shaped_parameter_on_an_element():
    """Test that a parameter shaped by a role bound to a single element is one issue at the parameter."""
    report = PassCompiler().prepare(_weighted(np.ones((4, 2)), [1.0, 2.0], source='single')).report_
    (issue,) = report.errors
    assert (issue.code, issue.path) == ('parameter-shape', 'input.W')
    assert 'group' in issue.fix


def test_role_element_reports_a_missing_state_a_role_declares():
    """Test that a declared read of a state the element lacks is reported with its path, role, state, and fix."""
    system = CompositeSystem(
        elements=[TankElement('tank')],
        interactions=[RateInteraction('growth', target='tank', state='n', rate=1.0)],
    )
    issues = _role_issues(system)
    assert len(issues) == 1
    assert issues[0].path == 'growth'
    assert "'target'" in issues[0].message
    assert "'n'" in issues[0].message
    assert 'level' in issues[0].message
    assert 'get_requirements()' in issues[0].fix


def test_role_element_reports_a_kind_the_role_does_not_accept():
    """Test that a role that acts on the agents of a group reports a single element."""
    system = CompositeSystem(
        elements=[TankElement('tank')],
        interactions=[WalkInteraction('walk', target='tank')],
        time=DiscreteTime(),
    )
    issues = _role_issues(system)
    assert len(issues) == 1
    assert 'accepts group, variable-group' in issues[0].message
    assert 'tank is of kind element' in issues[0].message


def test_role_element_reports_a_missing_state_no_role_declares():
    """Test that an undeclared read of a missing state gets the same issue, and no `AttributeError` reaches the user."""
    system = CompositeSystem(elements=[TankElement('tank')], interactions=[ReadingInteraction('read', target='tank')])
    issues = _role_issues(system)
    assert len(issues) == 1
    assert "'target'" in issues[0].message
    assert "'n'" in issues[0].message
    assert 'level' in issues[0].message
    with pytest.raises(RoleElementError, match=r"reads the state 'n'.*\[role-element\]"):
        PassCompiler().prepare(system).run()


def test_role_element_names_a_variadic_role():
    """Test that a missing state of an element of a variadic role names this role."""

    class TwoStatesElement(Element):
        """Element with the state the target of the interaction writes."""

        def __init__(self, label: str, x: Annotated[float, StateSpec()] = 1.0) -> None:
            super().__init__(label)
            self.x = x

    system = CompositeSystem(
        elements=[TwoStatesElement('first'), TankElement('tank')],
        interactions=[ReadingSourcesInteraction('read', target='first', sources=['tank'])],
    )
    issues = _role_issues(system)
    assert len(issues) == 1
    assert "'sources'" in issues[0].message
    assert "'missing'" in issues[0].message


def test_role_element_comes_with_the_other_problems():
    """Test that a role issue is one problem of the report, next to a problem of another component."""
    system = CompositeSystem(
        elements=[TankElement('tank')],
        interactions=[
            ReadingInteraction('read', target='tank'),
            RateInteraction('growth', target='tank', state='n', rate=1.0),
            RateInteraction('unbound', target='nowhere', state='n', rate=1.0),
        ],
    )
    issues = PassCompiler().prepare(system).report_.issues
    assert {issue.path for issue in issues if issue.code == 'role-element'} == {'read', 'growth'}
    assert any(issue.path == 'unbound' and issue.code == 'role' for issue in issues)


def test_role_element_keeps_a_missing_name_of_the_namespace():
    """Test that reading a name the namespace itself lacks is not reported as a state of a role."""

    class WrongNameInteraction(ReadingInteraction):
        """Interaction whose laws read a parameter that nothing declares."""

        def _get_laws(self, s):
            """Return a rate that reads a name of the namespace that does not exist."""
            return [RateLaw(s.target.level, s.undeclared)]

    system = CompositeSystem(elements=[TankElement('tank')], interactions=[WrongNameInteraction('read', target='tank')])
    with pytest.raises(AttributeError, match='undeclared'):
        PassCompiler().prepare(system)


@pytest.mark.parametrize('error', [ValueError('own error'), AttributeError('own error'), KeyError('own error')])
def test_role_element_keeps_the_other_errors_of_a_law(error):
    """Test that an error of a law that is not a missing attribute of a role reaches the user unchanged."""

    class FailingInteraction(ReadingInteraction):
        """Interaction whose laws raise an error of their own."""

        def _get_laws(self, s):
            """Raise the error of the test."""
            raise error

    system = CompositeSystem(elements=[TankElement('tank')], interactions=[FailingInteraction('read', target='tank')])
    with pytest.raises(type(error), match='own error') as caught:
        PassCompiler().prepare(system)
    assert caught.value is error


def test_role_element_leaves_a_valid_binding_alone():
    """Test that an element that carries what a role reads reports no role issue."""
    system = CompositeSystem(
        elements=[TankElement('tank')],
        interactions=[RateInteraction('growth', target='tank', state='level', rate=1.0)],
    )
    assert _role_issues(system) == []


def test_role_element_reports_a_state_that_a_function_reads():
    """Test that a function passed to a general interaction that reads a missing state gets the same issue."""
    growth = RateInteraction('growth', target='tank', state='level', rate=lambda s: s.target.missing)
    issues = _role_issues(CompositeSystem(elements=[TankElement('tank')], interactions=[growth]))
    assert len(issues) == 1
    assert "'target'" in issues[0].message
    assert "'missing'" in issues[0].message
