"""Test the `_defaults` module of `skcomplex.core.engine.solvers`."""

from skcomplex.core.engine.solvers import RewritingSolver
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.core.types import FormKind


def test_default_solvers_solve_their_form():
    """Test that the default solver of every form solves that form."""
    for form, solver in DEFAULT_SOLVERS.items():
        assert form == solver._get_form() or form in solver._get_forms()


def test_rewriting_form_runs_the_rewriting_solver():
    """Test that a problem of the rewriting form runs on the rewriting solver when no solver is passed."""
    assert DEFAULT_SOLVERS[FormKind.REWRITING] is RewritingSolver
