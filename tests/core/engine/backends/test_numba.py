"""Test the `_numba` module of `skcomplex.core.engine.backends`."""

import time

import pytest

from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import compile_system
from skcomplex.systems.physics.statistical_mechanics import ideal_gas


def _step_time(problem, repeats=20):
    """Return the mean time of one drift and boundary evaluation after a warm-up call."""
    x, p = problem.initial_state, problem.initial_parameters
    problem.drift(0.0, x, p)
    problem.boundary(0.0, x, p)
    start = time.perf_counter()
    for _ in range(repeats):
        problem.drift(0.0, x, p)
        problem.boundary(0.0, x, p)
    return (time.perf_counter() - start) / repeats


@pytest.mark.benchmark
def test_numba_is_three_times_faster_on_million_member_gas():
    """Test that one step of a million-member gas takes Numba at most a third of the NumPy time."""
    gas = ideal_gas(1_000_000, random_state=0)
    numpy_time = _step_time(compile_system(gas, backend=NumpyBackend()))
    numba_time = _step_time(compile_system(gas, backend=NumbaBackend()))
    assert numba_time <= numpy_time / 3
