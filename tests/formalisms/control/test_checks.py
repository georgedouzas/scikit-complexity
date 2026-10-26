"""Test the checks of the optimal control solvers against the system they run."""

import pytest

from skcomplex.core.base import SolverError, state
from skcomplex.core.components import Element
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.space import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.compile import validate
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.control import CollocationSolver, PontryaginSolver


class TankElement(Element):
    """Level that halves at every step."""

    level: float = state(1.0)

    def laws(self, s):
        """Return the update of the level."""
        return [UpdateLaw(s.level, s.level / 2)]


def _level(s):
    """Return the level of the tank."""
    return s.tank.level


def test_objective_issues_under_discrete_time():
    """Test that a collocation run with an objective on a system under discrete time raises a named error."""
    system = System(components=[TankElement('tank')], time=DiscreteTime())
    with pytest.raises(SolverError, match='objective of CollocationSolver needs static or continuous time'):
        Simulation(system, solver=CollocationSolver(_level)).run(until=1.0)


def test_objective_issues_name_the_subsystem_of_discrete_time():
    """Test that the issue of a subsystem under discrete time names the subsystem."""
    child = System('reservoir', components=[TankElement('tank')], time=DiscreteTime())
    issues = validate(System(components=[child]), solver=PontryaginSolver(_level)).errors
    issue = next(issue for issue in issues if issue.code == 'objective-time')
    assert issue.path == 'reservoir'
    assert issue.error is SolverError


def test_objective_issues_report_a_missing_objective(linear_quadratic):
    """Test that validation reports a missing objective as an error of the solver."""
    (issue,) = validate(linear_quadratic, solver=PontryaginSolver()).errors
    assert issue.code == 'objective-argument'
    assert issue.error is SolverError
    assert 'PontryaginSolver has no objective' in issue.message


def test_objective_issues_accept_a_valid_solver(linear_quadratic, quadratic_cost):
    """Test that a solver with an objective on a system under continuous time has no issues."""
    assert validate(linear_quadratic, solver=CollocationSolver(quadratic_cost)).ok
