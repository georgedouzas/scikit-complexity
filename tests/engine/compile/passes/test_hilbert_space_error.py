"""Test the `HilbertSpaceError` class."""

import pytest

from skcomplex.core.elements import GroupElement
from skcomplex.core.system import System
from skcomplex.engine.compile import (
    compile_system,
)
from skcomplex.engine.compile.passes import HilbertSpaceError
from skcomplex.engine.solvers import LindbladSolver
from skcomplex.models.physics.common.quantum import (
    SpinElement,
)


def test_compile_system_rejects_too_large_spaces():
    """Test that a space above the dimension limit raises an error naming the group, its dimension, and the limit."""
    system = System(elements=[GroupElement('chain', element=SpinElement, size=40)])
    with pytest.raises(HilbertSpaceError, match=r'chain of dimension 1099511627776.*max_dimension = 16777216'):
        compile_system(system, solver=LindbladSolver(max_dimension=2**24))
