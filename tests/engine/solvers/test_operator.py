"""Test the `OperatorSolver` class."""

import pytest

from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import LindbladSolver
from skcomplex.errors import SolverError


def test_operator_solvers_reject_classical_problems():
    """Test that an operator solver refuses a classical problem."""
    with pytest.raises(SolverError):
        LindbladSolver(random_state=0).check().prepare(compile_system(System()))
