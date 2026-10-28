"""Test the `QuantumProblemSpec` class."""

import math

from skcomplex.core.elements import LatticeElement
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import (
    compile_system,
)
from skcomplex.models.physics.common.quantum import (
    ModeElement,
    SpinElement,
    mode_space,
)
from skcomplex.specs import BaseProblemSpec, ProblemSpec, QuantumProblemSpec
from skcomplex.types import FormKind


def test_compile_system_gives_the_dimensions_of_a_chain_and_a_mode():
    """Test that a chain of spins and a mode compile to the tensor product of their spaces."""
    system = System(
        elements=[
            LatticeElement('chain', element=SpinElement, space=LatticeSpace(10)),
            ModeElement('cavity', hilbert=mode_space(4)),
        ],
    )
    problem = compile_system(system)
    assert isinstance(problem, QuantumProblemSpec)
    assert problem.form == FormKind.QUANTUM
    assert list(problem.dimensions) == [2] * 10 + [5]
    assert math.prod(problem.dimensions) == 5120


def test_quantum_problem_is_a_base_problem_and_not_the_compilers():
    """Test that a quantum problem derives from the problem base alone, and that its run is recorded."""
    assert issubclass(QuantumProblemSpec, BaseProblemSpec)
    assert not issubclass(QuantumProblemSpec, ProblemSpec)
