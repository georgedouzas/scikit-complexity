"""Test the `_linear_program_kernel` module of `skcomplex.core.engine.problems`."""

import numpy as np
import pytest
from scipy import sparse

from skcomplex.core.engine.enums import LinearProgramBound
from skcomplex.core.engine.problems._linear_program_kernel import LinearProgramKernel
from skcomplex.core.engine.problems._linear_program_spec import LinearProgramSideSpec, LinearProgramSpec
from skcomplex.core.engine.problems._reference_spec import ReferenceSpec
from skcomplex.core.errors import SolverError
from skcomplex.core.specs import EntrySpec, RunStateSpec


def _program(**settings):
    """Return a program that maximises one variable, recorded with its objective in a state vector of two values."""
    values = {
        'path': 'flows',
        'names': ('flows.v',),
        'variables': (np.array([0]),),
        'constraints': (),
        'relations': (),
        'matrix': sparse.csr_array((0, 1)),
        'lower': np.array([0.0]),
        'upper': np.array([3.0]),
        'row_lower': np.zeros(0),
        'row_upper': np.zeros(0),
        'costs': np.array([[-1.0]]),
        'signs': (-1.0,),
        'offsets': (0.0,),
        'objectives': (np.array([1]),),
    }
    return LinearProgramSpec(**{**values, **settings})


def _state(t=0.0):
    """Return a state of the program."""
    return RunStateSpec(t, np.zeros(2), np.zeros(1))


def test_run_keeps_one_solve_for_one_time_and_one_set_of_values():
    """Test that a second run at one time copies the values, and that a new time solves again."""
    kernel = LinearProgramKernel(_program()).prepare()
    assert kernel.run(_state()).values_.tolist() == [3.0, 3.0]
    kernel.run(_state())
    assert kernel.solves_ == 1
    kernel.run(_state(1.0))
    assert kernel.solves_ == 2


def test_run_raises_for_an_unbounded_program():
    """Test that an objective no bound limits raises an error that names the program."""
    kernel = LinearProgramKernel(_program(upper=np.array([np.inf]))).prepare()
    with pytest.raises(SolverError, match='flows'):
        kernel.run(_state())


def test_prepare_evaluates_the_bounds_at_a_state_without_solving():
    """Test that the bounds that compiled values give are set at a state, with no solve."""
    entry = EntrySpec('flows.limit', 0, (), None, None)
    reference = ReferenceSpec(entry, False, 0)
    program = _program(
        sides=(LinearProgramSideSpec(LinearProgramBound.COLUMN_UPPER, 0, 0),),
        function=lambda t, limit: [limit],
        arguments=(reference,),
    )
    kernel = LinearProgramKernel(program).prepare(RunStateSpec(0.0, np.zeros(2), np.array([2.0])))
    assert kernel.upper_.tolist() == [[2.0]]
    assert kernel.solves_ == 0
