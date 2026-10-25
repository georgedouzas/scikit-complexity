"""Test the `_solvers` module of `skcomplex.core.base`."""

import ast
import subprocess  # nosec B404: the test starts a fresh interpreter to see which modules a run loads
import sys

import numpy as np
import pytest

from skcomplex.core.base import Entry, Form, Layout, NumericDomain, SolverError, symbol
from skcomplex.core.problems import BaseProblem, BaseSolver

SCRIPT = '''
import sys
import numpy as np
from skcomplex.core.base import Entry, Form, Layout, NumericDomain, Supports, symbol
from skcomplex.core.problems import BaseProblem, BaseSolver


class HalvingSolver(BaseSolver[BaseProblem]):
    """Solver that halves the state once per unit of time."""

    form = Form.DISCRETE
    supports = Supports()

    def _advance(self, t, x, p, until):
        """Return the state halved once per unit of time."""
        return x * 0.5 ** (until - t)


problem = BaseProblem(
    Form.DISCRETE,
    Layout([Entry('a.x', 0, (), NumericDomain.REAL)]),
    Layout([]),
    np.array([8.0]),
    np.zeros(0),
    symbol('t'),
)
solver = HalvingSolver()
solver.prepare(problem, np.random.default_rng(0))
state = solver.run(0.0, problem.initial_state, problem.initial_parameters, 3.0)
print([float(state[0]), any(name.startswith('skcomplex.engine') for name in sys.modules)])
'''


def test_solver_advances_a_base_problem_without_the_engine():
    """Test that a solver written against the solver base advances a base problem and loads no engine module."""
    command = [sys.executable, '-c', SCRIPT]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603  # nosec B603
    assert ast.literal_eval(output) == [1.0, False]


def test_solver_rejects_a_problem_of_another_form():
    """Test that a solver rejects a problem whose form it does not solve, naming both forms."""

    class OdeSolver(BaseSolver[BaseProblem]):
        """Solver of ordinary differential equations."""

        form = Form.ODE

    problem = BaseProblem(Form.DISCRETE, Layout([]), Layout([]), None, None, symbol('t'))
    with pytest.raises(SolverError, match='discrete map'):
        OdeSolver().prepare(problem, None)


class _DoublingSolver(BaseSolver[BaseProblem]):
    """Solver that doubles the state once per unit of time and counts its releases."""

    form = Form.DISCRETE

    def _prepare(self, problem, rng, until=float('inf')):
        """Fit the base and start with no release."""
        super()._prepare(problem, rng, until)
        self.releases_ = 0

    def _advance(self, t, x, p, until):
        """Return the state doubled once per unit of time."""
        return x * 2.0 ** (until - t)

    def _release(self):
        """Count the release."""
        self.releases_ += 1


def _discrete_problem():
    """Return a discrete problem of one state with the initial value 1."""
    return BaseProblem(
        Form.DISCRETE,
        Layout([Entry('a.x', 0, (), NumericDomain.REAL)]),
        Layout([]),
        np.array([1.0]),
        np.zeros(0),
        symbol('t'),
    )


def test_solver_fit_sets_the_fitted_attributes():
    """Test that preparing a solver stores the problem, the generator, and the initial state, with nothing run yet."""
    problem, rng = _discrete_problem(), np.random.default_rng(0)
    solver = _DoublingSolver().prepare(problem, rng, 3.0)
    assert solver.problem_ is problem
    assert solver.rng_ is rng
    assert solver.initial_state_.tolist() == [1.0]
    assert solver.initial_state_ is not problem.initial_state
    assert (solver.events_, solver.stopped_at_, solver.snapshot_, dict(solver.agents_)) == ([], None, None, {})


def test_solver_run_advances_and_releases_on_the_final_run():
    """Test that `run` returns the advanced state and releases only on the final run."""
    problem = _discrete_problem()
    solver = _DoublingSolver().prepare(problem, np.random.default_rng(0))
    state = solver.run(0.0, problem.initial_state, problem.initial_parameters, 2.0)
    assert state.tolist() == [4.0]
    assert solver.releases_ == 0
    state = solver.run(2.0, state, problem.initial_parameters, 3.0, final=True)
    assert state.tolist() == [8.0]
    assert solver.releases_ == 1


def test_solver_run_of_no_length_returns_the_state():
    """Test that a run that does not go forward in time returns the state without advancing it."""
    problem = _discrete_problem()
    solver = _DoublingSolver().prepare(problem, np.random.default_rng(0))
    state = np.array([5.0])
    assert solver.run(1.0, state, problem.initial_parameters, 1.0, final=True) is state
    assert solver.releases_ == 1


def test_solver_speaks_only_the_estimator_verbs():
    """Test that the solver base has none of the former verbs."""
    for name in ('initialize', 'advance', 'plan', 'initial_state', 'drain_events', 'snapshot', 'agents_of'):
        assert not hasattr(BaseSolver, name)
