"""Test the general compiler: its settings, its report on a system, and the problem it runs to."""

from typing import Annotated

import networkx as nx
import pytest
from sage.all__sagemath_symbolics import SR, cos, identity_matrix, sin, sqrt

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import convolve, displacement, inverse_helmholtz, laplacian, tensor_divergence
from skcomplex.core.spaces import LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler, QuantumCompiler
from skcomplex.engine.problems import ProblemSpec
from skcomplex.engine.solvers import GillespieSolver
from skcomplex.errors import NotCheckedError, SolverError, ValidationError
from skcomplex.models.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.models.mathematics.fractals import RewritingCompiler
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


def test_general_compiler_names_the_compiler_of_a_quantum_system():
    """Test that the general compiler reports a quantum system with the compiler it needs, which compiles it."""
    system = System(elements=[QubitElement('q')])
    assert QuantumCompiler().check().prepare(system).report_.ok
    assert not RewritingCompiler().check().prepare(system).report_.ok
    compiler = PassCompiler().check().prepare(system)
    assert not compiler.report_.ok
    with pytest.raises(ValidationError, match=r'compiler=QuantumCompiler\(\)'):
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
