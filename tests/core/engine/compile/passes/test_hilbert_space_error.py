"""Test the `HilbertSpaceError` class."""

import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile.passes._hilbert_space_error import HilbertSpaceError
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.engine.solvers import LindbladSolver

from .....conftest import SpinElement


def test_compiler_rejects_too_large_spaces():
    """Test that a space above the dimension limit raises an error naming the group, its dimension, and the limit."""
    system = System(elements=[GroupElement('chain', element=SpinElement, size=40)])
    with pytest.raises(HilbertSpaceError, match=r'chain of dimension 1099511627776.*max_dimension = 16777216'):
        OperatorCompiler(solver=LindbladSolver(max_dimension=2**24)).prepare(system).run()
