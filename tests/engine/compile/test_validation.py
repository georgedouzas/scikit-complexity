"""Test the validation of systems against the requirements of their laws and the support of solvers and backends."""

import re
from typing import Annotated

import networkx as nx
import pytest
from sage.all__sagemath_gap import libgap
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import Element, GroupElement, LatticeElement
from skcomplex.core.interactions import HyperedgeInteraction, Interaction, NeighborInteraction
from skcomplex.core.laws import NoiseLaw, RateLaw, RuleLaw, SampleLaw, UpdateLaw
from skcomplex.core.laws.utils import mean, neighbors
from skcomplex.core.neighborhoods import MemoryNeighborhood
from skcomplex.core.spaces import CayleySpace, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.core.times import DiscreteTime
from skcomplex.engine.compile import PassCompiler, compile_system
from skcomplex.engine.solvers import LammpsSolver, MilsteinSolver, PontryaginSolver, RK4Solver
from skcomplex.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.errors import ValidationError, ValidationWarning
from skcomplex.specs import (
    EnergySpec,
    IssueSpec,
    MoveSpec,
    ObservableSpec,
    ParameterSpec,
    RoleSpec,
    SamplerSpec,
    StateSpec,
    ValidationReportSpec,
    VisualSpec,
    WienerProcessSpec,
)
from skcomplex.types import FormKind, NoiseKind

from ...conftest import HoldingElement, NodeElement, _correlated
from .conftest import PairElement


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

    def walk(self, agent, model):
        """Count one step."""
        agent.steps += 1


def _codes(report):
    """Return the codes of the issues of a report."""
    return [issue.code for issue in report.issues]


def test_valid_system_has_no_issues():
    """Test that a system of stocks with growth validates without issues."""
    system = System(
        elements=[StockElement('rabbits')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    report = PassCompiler().check().prepare(system).report_
    assert report.ok
    assert report.issues == ()


def test_several_errors_are_reported_at_once():
    """Test that an unbound role and a vector state without a space are both reported."""
    system = System(
        elements=[StockElement('rabbits'), WalkerElement('walker')],
        interactions=[GrowthInteraction('growth', target='foxes')],
    )
    report = PassCompiler().check().prepare(system).report_
    assert sorted(_codes(report)) == ['role', 'space-missing']
    assert {issue.path for issue in report.errors} == {'growth', 'walker'}
    assert all(issue.fix for issue in report.issues)


def test_unsupported_solver_lists_capable_solvers():
    """Test that a variable group validated with RK4 reports the unsupported components and capable solvers."""
    system = System(elements=[GroupElement('ants', element=AntElement, size=3, variable=True)])
    report = PassCompiler(solver=RK4Solver()).check().prepare(system).report_
    issue = next(issue for issue in report.errors if 'laws' in issue.message)
    assert issue.code == 'solver-unsupported'
    assert 'RuleLaw' in issue.message
    assert 'RuleLaw' in issue.fix


def test_mixed_laws_suggest_partitions():
    """Test that rates and updates in one system without children are reported with both components."""
    system = System(
        elements=[StockElement('rabbits'), CounterElement('clock')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    (issue,) = PassCompiler().check().prepare(system).report_.errors
    assert issue.code == 'laws-mixed'
    assert 'RateLaw: rabbits' in issue.message
    assert 'UpdateLaw: clock' in issue.message
    assert 'nested system' in issue.fix


def test_mixed_laws_in_nested_systems_are_valid():
    """Test that rates and updates in sibling nested systems validate as coupled partitions."""
    system = System(
        systems=[
            System(
                'meadow',
                elements=[StockElement('rabbits')],
                interactions=[GrowthInteraction('growth', target='rabbits')],
            ),
            System('clock', elements=[CounterElement('counter')]),
        ],
    )
    assert PassCompiler().check().prepare(system).report_.ok


def test_report_raises_every_error():
    """Test that checking a report raises every error with its path and code, and emits its warnings."""
    report = ValidationReportSpec(
        (
            IssueSpec('role', 'growth', 'The role is unbound.', 'Bind it.'),
            IssueSpec('law-domain', 'drift', 'The law targets a categorical state.', 'Use a jump law.'),
            IssueSpec('note', 'system', 'Something to know.', 'Nothing to do.', severity='warning'),
        ),
    )
    with pytest.warns(ValidationWarning, match='Something to know'), pytest.raises(ValidationError) as error:
        report.check()
    assert '2 error(s)' in str(error.value)
    assert 'growth: The role is unbound. Bind it. [role]' in str(error.value)
    assert '[law-domain]' in str(error.value)


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


def test_validate_control_outside_bounds():
    """Test that an adaptive parameter starting outside its bounds is one error at the parameter, naming its bounds."""
    (issue,) = PassCompiler().check().prepare(System(elements=[PlantElement('plant', u=2.0)])).report_.errors
    assert issue.path == 'plant.u'
    assert issue.code == 'parameter-bounds'
    # typing caches equal annotations, so the bounds print as those of the first equal spec: (0, 1) or (0.0, 1.0)
    assert re.search(r'\(0(\.0)?, 1(\.0)?\)', issue.fix)


def test_validate_controls_with_an_optimal_control_solver():
    """Test that a system with bounded controls is valid for an optimal control solver and for a stepper."""
    solver = PontryaginSolver(objective=lambda s: s.plant.x**2)
    assert PassCompiler(solver=solver).check().prepare(System(elements=[PlantElement('plant')])).report_.ok
    assert PassCompiler(solver=RK4Solver()).check().prepare(System(elements=[PlantElement('plant')])).report_.ok


def test_validate_noise_with_an_optimal_control_solver():
    """Test that a system with a noise law validated with Pontryagin reports that the solver lacks noise laws."""
    solver = PontryaginSolver(objective=lambda s: s.plant.x**2)
    issues = PassCompiler(solver=solver).check().prepare(System(elements=[NoisyPlantElement('plant')])).report_.errors
    issue = next(issue for issue in issues if issue.code == 'solver-unsupported')
    assert 'PontryaginSolver does not support the laws NoiseLaw' in issue.message


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
        energy = EnergySpec(-s.spin * sum(neighbors(s.spin, memory, reduce='each')), memory)
        return [SampleLaw(energy, MoveSpec(s.spin), SamplerSpec('glauber', 2.0))]


def test_group_space_reports_the_checkerboard_schedule():
    """Test that a sample with the checkerboard schedule on a group space is one issue naming the other schedules."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2)
    system = System(elements=[LatticeElement('spins', element=GroupSpinElement, space=space)])
    report = PassCompiler().check().prepare(system).report_
    issues = report.issues
    (issue,) = [issue for issue in issues if issue.code == 'schedule-unsupported']
    assert issue.path == 'spins'
    assert "'synchronous'" in issue.fix
    assert "'random-sequential'" in issue.fix


def test_group_space_reports_an_image_visual():
    """Test that an image visual on a group space is reported with the Cayley visual as its fix."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2)
    group = LatticeElement('spins', element=GroupSpinElement, space=space, visual=VisualSpec('image'))
    system = System(elements=[group])
    report = PassCompiler().check().prepare(system).report_
    issue = next(issue for issue in report.issues if issue.code == 'visual-unsupported')
    assert "'cayley'" in issue.fix


def test_group_space_reports_a_ball_above_the_site_limit():
    """Test that a ball of the free group of rank 3 with more sites than the limit is reported."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b', 'c']), ball=12, max_sites=100)
    issues = PassCompiler().check().prepare(System(space=space)).report_.issues
    assert [issue.code for issue in issues] == ['space-too-large']
    assert 'more than 100 sites' in issues[0].message


def test_lattice_with_a_bad_boundary_is_reported():
    """Test that a lattice whose boundary is not supported is a space-invalid error that lists the boundaries."""
    space = LatticeSpace((4, 4), boundary='spherical')
    (issue,) = PassCompiler().check().prepare(System(elements=[StockElement('rabbits')], space=space)).report_.errors
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
    assert {issue.code for issue in PassCompiler().check().prepare(system).report_.errors} == {'space-invalid', 'role'}
    with pytest.raises(ValidationError, match='space-invalid'):
        compile_system(system)


def test_validation_keeps_the_planned_compilation():
    """Test that validation returns the compilation state planned once, which holds the nodes and the plan."""
    system = System(
        elements=[StockElement('rabbits')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
    )
    compilation = PassCompiler().check().prepare(system).compilation_
    assert [node.path for node in compilation.nodes] == ['rabbits', 'growth']
    assert [partition.name for partition in compilation.plan.partitions] == [system.label]


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


def test_validate_reports_mixed_conventions():
    """Test that laws of different conventions on one process are reported with both laws and the process."""
    system = System(elements=[PairElement('pair', conventions=('ito', 'stratonovich'))])
    report = PassCompiler().check().prepare(system).report_
    issues = report.issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-convention-mixed']
    assert 'ito NoiseLaw of pair' in issue.message
    assert 'stratonovich NoiseLaw of pair' in issue.message
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
    report = PassCompiler().check().prepare(system).report_
    (issue,) = [issue for issue in report.issues if issue.code == 'noise-correction-scope']
    assert 'NoiseLaw of coupled' in issue.message
    assert 'NoiseLaw of single' in issue.message
    assert 'Itô' in issue.fix


def test_validate_reports_invalid_covariance():
    """Test that a covariance with a negative eigenvalue is reported with the element and the eigenvalue."""
    (issue,) = PassCompiler().check().prepare(_correlated([[1.0, 2.0], [2.0, 1.0]])).report_.issues
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
    report = PassCompiler().check().prepare(system).report_
    assert [issue.code for issue in report.issues] == ['wiener-covariance']


def test_validate_reports_handle_as_value():
    """Test that a Wiener handle used inside a rate law is reported with the law."""
    system = System(
        elements=[HoldingElement('a')],
        interactions=[MisuseInteraction('misuse', target='a', noise=WienerProcessSpec('market'))],
    )
    report = PassCompiler().check().prepare(system).report_
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

    system = System(
        elements=[CounterElement('counter')],
        interactions=[SingleInteraction('single', target='counter', noise=WienerProcessSpec('market'))],
        time=DiscreteTime(),
    )
    (issue,) = [issue for issue in PassCompiler().check().prepare(system).report_.issues if issue.code == 'wiener-time']
    assert issue.path == 'market'
    assert 'continuous time' in issue.fix


def test_validate_reports_milstein_with_shared_process():
    """Test that Milstein with a shared process gives one error that names the law and suggests other solvers."""
    system = System(elements=[PairElement('pair')])
    report = PassCompiler(solver=MilsteinSolver()).check().prepare(system).report_
    (issue,) = report.issues
    assert issue.code == 'noise-unsupported'
    assert 'NoiseLaw of pair' in issue.message
    assert 'MilsteinSolver' in issue.message
    assert 'shared noise' in issue.fix


def test_validate_reports_lammps_with_shared_process():
    """Test that the LAMMPS backend with a shared process is reported with the backends that support it."""
    issues = PassCompiler(solver=LammpsSolver()).check().prepare(System(elements=[PairElement('pair')])).report_.issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-unsupported']
    assert 'NoiseLaw of pair' in issue.message
    assert 'LammpsSolver' in issue.message


def test_validate_accepts_correlated_noise_with_automatic_solver():
    """Test that the automatic SDE solver declares correlated noise, so validation accepts it."""
    assert PassCompiler().check().prepare(_correlated([[1.0, 0.5], [0.5, 1.0]])).report_.ok
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

    (issue,) = PassCompiler().check().prepare(System(elements=[MixedElement('tank')])).report_.errors
    assert issue.path == 'tank'
    assert '`_get_laws` of tank must return a list of' in issue.message
    with pytest.raises(ValidationError, match='tank'):
        compile_system(System(elements=[MixedElement('tank')]))


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
        return [RateLaw(s.nodes.x, mean(s.nodes.x) - s.nodes.x)]


def _bindings(interaction, space):
    """Return the binding issues of a group of four nodes on a space, coupled by an interaction."""
    nodes = GroupElement('nodes', element=NodeElement, size=4)
    system = System(elements=[nodes], interactions=[interaction], space=space)
    report = PassCompiler().check().prepare(system).report_
    issues = report.issues
    return [issue for issue in issues if issue.code == 'binding']


def test_neighbor_interaction_without_neighbours_takes_no_laws_per_edge():
    """Test that an interaction over neighbours of members on no lattice or network binds but has no laws per edge."""
    assert _bindings(SpreadInteraction('spread', nodes='nodes'), None) == []
    nodes = GroupElement('nodes', element=NodeElement, size=4)
    with pytest.raises(ValidationError, match='laws per edge'):
        compile_system(System(elements=[nodes], interactions=[SpreadInteraction('spread', nodes='nodes')]))


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
