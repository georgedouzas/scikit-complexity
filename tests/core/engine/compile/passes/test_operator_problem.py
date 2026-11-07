"""Test the `OperatorProblemSpec` class."""

import math

from skcomplex.core.components.elements import LatticeElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import OperatorCompiler
from skcomplex.core.engine.problems import BaseProblemSpec, OperatorProblemSpec, ProblemSpec
from skcomplex.core.types import FormKind
from skcomplex.systems.physics.common.quantum import ModeElement, mode_space

from .....conftest import SpinElement


def test_compiler_gives_the_dimensions_of_a_chain_and_a_mode():
    """Test that a chain of spins and a mode compile to the tensor product of their spaces."""
    system = System(
        elements=[
            LatticeElement('chain', element=SpinElement, space=LatticeSpace(10)),
            ModeElement('cavity', hilbert=mode_space(4)),
        ],
    )
    problem = OperatorCompiler().prepare(system).run().problem_
    assert isinstance(problem, OperatorProblemSpec)
    assert problem.form == FormKind.OPERATOR
    assert list(problem.dimensions) == [2] * 10 + [5]
    assert math.prod(problem.dimensions) == 5120


def test_operator_problem_is_a_base_problem_and_not_the_compilers():
    """Test that an operator problem derives from the problem base alone, and that its run is recorded."""
    assert issubclass(OperatorProblemSpec, BaseProblemSpec)
    assert not issubclass(OperatorProblemSpec, ProblemSpec)
