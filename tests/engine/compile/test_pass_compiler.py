"""Test the general compiler: its settings, its report on a system, and the problem it runs to."""

import pytest

from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler, QuantumCompiler
from skcomplex.engine.problems import ProblemSpec
from skcomplex.engine.solvers import GillespieSolver
from skcomplex.errors import NotCheckedError, SolverError, ValidationError
from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationElement
from skcomplex.models.mathematics.fractals import RewritingCompiler
from skcomplex.models.physics.common.quantum import QubitElement
from skcomplex.specs import ObservableSpec, ValidationReportSpec


def _growth():
    """Return a system of one growing population."""
    return System(
        elements=[PopulationElement('rabbits', n=10)],
        interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.5)],
    )


def test_compiler_prepares_its_report_and_runs_to_the_problem():
    """Test that `prepare` sets the report of a system and `run` returns its problem."""
    compiler = PassCompiler().check().prepare(_growth())
    assert isinstance(compiler.report_, ValidationReportSpec)
    assert compiler.report_.ok
    problem = compiler.run({'rabbits.n': 20})
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
