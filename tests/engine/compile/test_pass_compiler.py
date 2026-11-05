"""Test the general compiler: its settings, its report on a system, and the problem it runs to."""

from typing import Annotated

import networkx as nx
import pytest
from sage.all__sagemath_symbolics import SR, cos, identity_matrix, sin, sqrt

from skcomplex.core.elements import Element, FieldElement, GroupElement
from skcomplex.core.interactions import Interaction, RateInteraction, SamplingInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import (
    codifferential,
    convolve,
    displacement,
    exterior_derivative,
    hodge_star,
    inverse_helmholtz,
    laplacian,
    neighbors,
    tensor_divergence,
)
from skcomplex.core.spaces import (
    CellComplexSpace,
    CofaceNeighborhood,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    NetworkSpace,
)
from skcomplex.core.system import System
from skcomplex.engine.compile import OperatorCompiler, PassCompiler, RewritingCompiler
from skcomplex.engine.problems import ProblemSpec
from skcomplex.engine.solvers import GillespieSolver
from skcomplex.errors import NotCheckedError, SolverError, ValidationError
from skcomplex.models.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.models.physics.common.quantum import QubitElement
from skcomplex.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, ValidationReportSpec, VisualSpec


def _growth():
    """Return a system of one growing population."""
    return System(
        elements=[PopulationElement('rabbits', n=10)],
        interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.5)],
    )


def test_compiler_prepares_its_report_and_runs_to_the_problem():
    """Test that `prepare` sets the report of a system and `run` sets its problem."""
    compiler = PassCompiler().check().prepare(_growth())
    assert isinstance(compiler.report_, ValidationReportSpec)
    assert compiler.report_.ok
    problem = compiler.run({'rabbits.n': 20}).problem_
    assert isinstance(problem, ProblemSpec)
    assert problem.initial_state.tolist() == [20.0]


def test_compiler_run_raises_the_errors_of_its_report():
    """Test that `run` raises the errors that `prepare` found, after `prepare` reported them."""
    compiler = PassCompiler(solver=GillespieSolver()).check().prepare(_growth())
    assert [issue.code for issue in compiler.report_.errors] == ['solver-unsupported']
    with pytest.raises(SolverError, match='solver-unsupported'):
        compiler.run()


def test_compiler_needs_a_check_before_prepare():
    """Test that a compiler that was not checked refuses to prepare."""
    with pytest.raises(NotCheckedError):
        PassCompiler().prepare(_growth())


@pytest.mark.parametrize(
    'settings',
    [{'partition': 'members'}, {'passes': [object()]}, {'backend': 'numpy'}, {'observables': ['x']}],
)
def test_compiler_check_rejects_invalid_settings(settings):
    """Test that `check` rejects a partition, passes, a backend, or observables of the wrong kind."""
    with pytest.raises(ValidationError):
        PassCompiler(**settings).check()


def test_compiler_rejects_a_target_that_is_not_a_system():
    """Test that a compiler prepares only on a system."""
    with pytest.raises(ValidationError, match='prepares on a system'):
        PassCompiler().check().prepare(object())


def test_general_compiler_names_the_compiler_of_a_operator_system():
    """Test that the general compiler reports an operator system with the compiler it needs, which compiles it."""
    system = System(elements=[QubitElement('q')])
    assert OperatorCompiler().check().prepare(system).report_.ok
    assert not RewritingCompiler().check().prepare(system).report_.ok
    compiler = PassCompiler().check().prepare(system)
    assert not compiler.report_.ok
    with pytest.raises(ValidationError, match=r'compiler=OperatorCompiler\(\)'):
        compiler.run()


def test_compiler_reports_an_observable_of_a_missing_system():
    """Test that an observable labelled with a nested system that does not exist is an issue of the report."""
    observable = ObservableSpec('village/size', lambda s: s.rabbits.n)
    report = PassCompiler(observables=[observable]).check().prepare(_growth()).report_
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
        report = PassCompiler().check().prepare(System(elements=[field])).report_
        assert [(issue.code, issue.path) for issue in report.errors] == [('screening', 'm')]
        assert 'screening' in report.errors[0].message


def test_compiler_reports_a_kernel_that_reads_a_state():
    """Test that a convolution kernel that reads a state is a `kernel-state` issue naming the state."""
    field = _nonlocal_field('m', lambda s: convolve(s.u, s.u * displacement(0)), LatticeSpace(8))
    report = PassCompiler().check().prepare(System(elements=[field])).report_
    assert [(issue.code, issue.path) for issue in report.errors] == [('kernel-state', 'm')]
    assert 'u_m' in report.errors[0].message


def test_compiler_reports_a_convolution_on_a_network():
    """Test that a convolution on a network is a `law-space` issue saying it needs a lattice space."""
    field = _nonlocal_field('net', lambda s: convolve(s.u, displacement(0)), NetworkSpace(nx.path_graph(4)))
    report = PassCompiler().check().prepare(System(elements=[field])).report_
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
    report = PassCompiler().check().prepare(System(elements=[field])).report_
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
    system = System(elements=[field, other], interactions=[_CrossInteraction('cross', first='a', second='b')])
    report = PassCompiler().check().prepare(system).report_
    assert [issue.code for issue in report.errors] == ['operator-spaces']


def test_compiler_raises_every_nonlocal_issue_at_once():
    """Test that a convolution on a network and a negative screening raise one error that lists both laws."""
    network = _nonlocal_field('net', lambda s: convolve(s.u, displacement(0)), NetworkSpace(nx.path_graph(4)))
    lattice = _nonlocal_field('grid', lambda s: inverse_helmholtz(s.u, sqrt(s.k)), LatticeSpace(8))
    compiler = PassCompiler().check().prepare(System(elements=[network, lattice]))
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
    return PassCompiler().check().prepare(System(elements=[field])).report_.issues


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
    issues = PassCompiler().check().prepare(System(elements=[field])).report_.issues
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
    return PassCompiler().check().prepare(system).report_


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
    system = System(elements=_cell_groups(), interactions=interactions, time=ContinuousTime())
    issues = [issue for issue in _report_of(system).issues if issue.code == 'law-degree']
    assert sorted(issue.path for issue in issues) == ['down', 'sample']
    assert 'exterior_derivative' in issues[0].fix
    assert 'edges' in issues[1].fix


def test_report_holds_a_degree_issue_for_an_exterior_derivative_of_the_highest_degree():
    """Test that an exterior derivative of a hyperedge state on a complex of dimension 2 is reported."""
    rate = RateInteraction('up', target='faces', state='f', rate=lambda n: exterior_derivative(n.target.f))
    issues = _report_of(System(elements=_cell_groups(), interactions=[rate])).issues
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
        issues = _report_of(System(elements=_cell_groups(), interactions=[interaction])).issues
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
    issues = _report_of(System(elements=_cell_groups(), interactions=interactions)).issues
    assert not [issue for issue in issues if issue.code == 'law-degree']


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
    issues = _report_of(System(elements=[sites], interactions=[rate])).issues
    assert [issue.code for issue in issues if issue.path == 'rate'] == ['law-space']
    issues = _report_of(System(elements=[sites], interactions=[sampling], time=DiscreteTime())).issues
    assert [issue.code for issue in issues if issue.path == 'sample'] == ['law-space']


def test_report_holds_a_space_issue_for_the_exterior_calculus_outside_a_cell_complex():
    """Test that an exterior derivative of a state of a plain network is reported as a law-space issue."""
    sites = GroupElement('sites', element=_CellSiteElement, space=NetworkSpace(nx.cycle_graph(4)), on='sites')
    rate = RateInteraction('rate', target='sites', state='u', rate=lambda n: exterior_derivative(n.target.u))
    issues = _report_of(System(elements=[sites], interactions=[rate])).issues
    assert [issue.code for issue in issues if issue.path == 'rate'] == ['law-space']
