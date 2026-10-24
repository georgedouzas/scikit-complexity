"""Share the helper of the synchronization model tests."""

import math

import numpy as np
import numpy.typing as npt

from skcomplex.core.compile import Problem, compile_system
from skcomplex.core.system import System


def _random_drift(system: System) -> tuple[Problem, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the phases at a random state and the drift of a phase model there."""
    problem = compile_system(system)
    state = problem.initial_state.copy()
    state[:] = np.random.default_rng(3).uniform(0, 2 * math.pi, state.size)
    return problem, state, problem.drift(0.0, state, problem.initial_parameters)
