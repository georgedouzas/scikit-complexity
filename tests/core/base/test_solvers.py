"""Test the `_solvers` module of `skcomplex.core.base`."""

import ast
import subprocess  # nosec B404: the test starts a fresh interpreter to see which modules a run loads
import sys

import pytest

from skcomplex.core.base import BaseProblem, BaseSolver, Form, Layout, SolverError, symbol

SCRIPT = '''
import sys
import numpy as np
from skcomplex.core.base import BaseProblem, BaseSolver, Entry, Form, Layout, NumericDomain, Supports, symbol


class HalvingSolver(BaseSolver[BaseProblem]):
    """Solver that halves the state once per unit of time."""

    form = Form.DISCRETE
    supports = Supports()

    def advance(self, t, x, p, until):
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
solver.initialize(problem, np.random.default_rng(0))
state = solver.advance(0.0, problem.initial_state, problem.initial_parameters, 3.0)
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

    problem = BaseProblem(Form.DISCRETE, Layout([]), Layout([]), None, None, symbol('t'))  # type: ignore[arg-type]
    with pytest.raises(SolverError, match='discrete map'):
        OdeSolver().initialize(problem, None)  # type: ignore[arg-type]
