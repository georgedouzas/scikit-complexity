"""Test the `QuantumProblem` class."""

import math

from skcomplex.core.elements import LatticeElement
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import (
    compile_system,
)
from skcomplex.engine.problems import BaseProblem, Form, Problem, QuantumProblem
from skcomplex.models.physics.common.quantum import (
    ModeElement,
    SpinElement,
    mode_space,
)


def test_compile_system_gives_the_dimensions_of_a_chain_and_a_mode():
    """Test that a chain of spins and a mode compile to the tensor product of their spaces."""
    system = System(
        elements=[
            LatticeElement('chain', element=SpinElement, space=LatticeSpace(10)),
            ModeElement('cavity', hilbert=mode_space(4)),
        ],
    )
    problem = compile_system(system)
    assert isinstance(problem, QuantumProblem)
    assert problem.form == Form.QUANTUM
    assert list(problem.dimensions) == [2] * 10 + [5]
    assert math.prod(problem.dimensions) == 5120


def test_quantum_problem_is_a_base_problem_and_not_the_compilers():
    """Test that a quantum problem derives from the problem base alone, and that its run is recorded."""
    assert issubclass(QuantumProblem, BaseProblem)
    assert not issubclass(QuantumProblem, Problem)
