"""Test the `_base` module of `skcomplex.engine.solvers`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import RR

from skcomplex.core.errors import SolverError
from skcomplex.engine._utils import symbol
from skcomplex.engine.problems import BaseProblem, Entry, Form, Layout
from skcomplex.engine.solvers import BaseSolver


def test_solver_rejects_a_problem_of_another_form():
    """Test that a solver rejects a problem whose form it does not solve, naming both forms."""

    class OdeSolver(BaseSolver[BaseProblem]):
        """Solver of ordinary differential equations."""

        form = Form.ODE

    problem = BaseProblem(Form.DISCRETE, Layout(()), Layout(()), None, None, symbol('t'))
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
        Layout((Entry('a.x', 0, (), RR),)),
        Layout(()),
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
