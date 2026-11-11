"""Test the `_base` module of `skcomplex.core.engine.solvers`."""

import inspect

import numpy as np
import pytest
from sage.all__sagemath_symbolics import RR

from skcomplex.core.engine import solvers
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.problems import BaseProblemSpec
from skcomplex.core.engine.solvers import (
    AgentView,
    BaseSolver,
    FractionalABMSolver,
    GillespieSolver,
    GrunwaldLetnikovSolver,
    L1Solver,
    LatticeSolver,
    MonteCarloSolver,
    ScipyODESolver,
    Solver,
    SystemView,
)
from skcomplex.core.enums import FormKind, OrderKind
from skcomplex.core.errors import SkcomplexError, SolverError
from skcomplex.core.specs import BaseSpec, EntrySpec, LayoutSpec, StepSpec

SOLVERS = [
    item
    for item in vars(solvers).values()
    if inspect.isclass(item)
    and issubclass(item, BaseSolver)
    and not inspect.isabstract(item)
    and item not in {BaseSolver, Solver}
]
VIEWS = {AgentView, SystemView}


def test_solver_rejects_a_problem_of_another_form():
    """Test that a solver rejects a problem whose form it does not solve, naming both forms."""

    class OdeSolver(BaseSolver[BaseProblemSpec]):
        """Solver of ordinary differential equations."""

        @classmethod
        def _get_form(cls):
            return FormKind.ODE

    problem = BaseProblemSpec(FormKind.DISCRETE, LayoutSpec(()), LayoutSpec(()), None, None, symbol('t'))
    with pytest.raises(SolverError, match='discrete map'):
        OdeSolver().prepare(problem)


class _DoublingSolver(BaseSolver[BaseProblemSpec]):
    """Solver that doubles the state once per unit of time and counts its releases."""

    @classmethod
    def _get_form(cls):
        """Return the discrete map form."""
        return FormKind.DISCRETE

    def _prepare(self, context):
        """Fit the base and start with no release."""
        super()._prepare(context)
        self.releases_ = 0

    def _advance(self, t, x, p, until):
        """Return the state doubled once per unit of time."""
        return x * 2.0 ** (until - t)

    def _release(self):
        """Count the release."""
        self.releases_ += 1


def _discrete_problem():
    """Return a discrete problem of one state with the initial value 1."""
    return BaseProblemSpec(
        FormKind.DISCRETE,
        LayoutSpec((EntrySpec('a.x', 0, (), RR),)),
        LayoutSpec(()),
        np.array([1.0]),
        np.zeros(0),
        symbol('t'),
    )


def test_solver_fit_sets_the_fitted_attributes():
    """Test that preparing a solver stores the problem, the generator, and the initial state, with nothing run yet."""
    problem, rng = _discrete_problem(), np.random.default_rng(0)
    solver = _DoublingSolver().set_params(random_state=rng, until=3.0).prepare(problem)
    assert solver.problem_ is problem
    assert solver.rng_ is rng
    assert solver.initial_state_.tolist() == [1.0]
    assert solver.initial_state_ is not problem.initial_state
    assert (solver.events_, solver.stopped_at_, solver.snapshot_, dict(solver.agents_)) == ([], None, None, {})


def test_solver_run_advances_and_releases_on_the_final_run():
    """Test that `run` keeps the advanced state in `state_` and releases only on the final run."""
    problem = _discrete_problem()
    solver = _DoublingSolver(random_state=0).prepare(problem)
    state = solver.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0)).state_
    assert state.tolist() == [4.0]
    assert solver.releases_ == 0
    state = solver.run(StepSpec(2.0, state, problem.initial_parameters, 3.0), final=True).state_
    assert state.tolist() == [8.0]
    assert solver.releases_ == 1


def test_solver_run_of_no_length_returns_the_state():
    """Test that a run that does not go forward in time keeps the state without advancing it."""
    problem = _discrete_problem()
    solver = _DoublingSolver(random_state=0).prepare(problem)
    state = np.array([5.0])
    assert solver.run(StepSpec(1.0, state, problem.initial_parameters, 1.0), final=True).state_ is state
    assert solver.releases_ == 1


def test_solver_speaks_only_the_estimator_verbs():
    """Test that the solver base has none of the former main public methods."""
    for name in ('initialize', 'advance', 'plan', 'initial_state', 'drain_events', 'snapshot', 'agents_of'):
        assert not hasattr(BaseSolver, name)


def test_solvers_support_the_tags_of_the_classes_they_support():
    """Test that a solver supports the tags of its supported classes and of their subclasses, and no other law kind."""
    ode = ScipyODESolver.get_supported_tags()
    assert ode.time == {'continuous'}
    assert 'jump' not in ode.laws
    assert {'none', 'lattice', 'phase space', 'hypergraph'} <= ode.space
    assert GillespieSolver.get_supported_tags().laws == {'jump', 'update'}
    assert 'none' not in LatticeSolver.get_supported_tags().space
    assert MonteCarloSolver.get_supported_tags().laws == {'sampling'}


@pytest.mark.parametrize('solver', SOLVERS, ids=lambda solver: solver.__name__)
def test_every_solver_declares_orders(solver):
    """Test that every solver declares the orders it integrates."""
    assert solver._get_support().orders


@pytest.mark.parametrize('solver', [GrunwaldLetnikovSolver, L1Solver, FractionalABMSolver])
def test_fractional_solvers_support_every_order(solver):
    """Test that the fractional solvers integrate integer and fractional orders."""
    assert solver._get_support().orders == set(OrderKind)


def test_public_names_are_classes_under_the_bases():
    """Every public name is a solver, a value a solver returns, an error, or a view that system rules receive."""
    for name in solvers.__all__:
        item = getattr(solvers, name)
        assert inspect.isclass(item), name
        assert issubclass(item, BaseSolver | BaseSpec | SkcomplexError) or item in VIEWS, name


def test_concrete_solvers_answer_their_hooks():
    """Every concrete solver names its form and what it supports, and takes a random state and an end time."""
    for name in solvers.__all__:
        item = getattr(solvers, name)
        if not inspect.isclass(item) or not issubclass(item, BaseSolver) or inspect.isabstract(item):
            continue
        try:
            item._get_form()
            item._get_support()
        except NotImplementedError:
            continue
        assert {'random_state', 'until'} <= set(item().get_params()), name
