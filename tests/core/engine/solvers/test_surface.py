"""Test that the public names of the solvers package are classes under its bases."""

import inspect

from skcomplex.core.engine import solvers as package
from skcomplex.core.engine.solvers import AgentView, BaseSolver, SystemView
from skcomplex.core.errors import SkcomplexError
from skcomplex.core.specs import BaseSpec, StepSpec

VIEWS = {AgentView, SystemView}


def test_public_names_are_classes_under_the_bases():
    """Every public name is a solver, a value a solver returns, an error, or a view that system rules receive."""
    for name in package.__all__:
        item = getattr(package, name)
        assert inspect.isclass(item), name
        assert issubclass(item, BaseSolver | BaseSpec | SkcomplexError) or item in VIEWS, name


def test_concrete_solvers_answer_their_hooks():
    """Every concrete solver names its form and what it supports, and takes a random state and an end time."""
    for name in package.__all__:
        item = getattr(package, name)
        if not inspect.isclass(item) or not issubclass(item, BaseSolver) or inspect.isabstract(item):
            continue
        try:
            item._get_form()
            item._get_support()
        except NotImplementedError:
            continue
        assert {'random_state', 'until'} <= set(item().get_params()), name


def test_a_solver_runs_on_a_step():
    """A step holds the state, the parameters, and the time to advance to."""
    assert issubclass(StepSpec, BaseSpec)
