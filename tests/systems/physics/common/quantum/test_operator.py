"""Test the `OperatorSolver` class."""

import pytest

from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.quantum import LindbladSolver


def test_operator_solvers_reject_classical_problems():
    """Test that an operator solver refuses a classical problem."""
    with pytest.raises(SolverError):
        LindbladSolver(random_state=0).prepare(PassCompiler().prepare(System()).run().problem_)
