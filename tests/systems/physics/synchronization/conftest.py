"""Share the helper of the synchronization system tests."""

import math

import numpy as np
import numpy.typing as npt

from skcomplex.core.components.system import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems import ProblemSpec


def _random_drift(system: System) -> tuple[ProblemSpec, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the phases at a random state and the drift of a phase system there."""
    problem = PassCompiler().prepare(system).run().problem_
    assert isinstance(problem, ProblemSpec)
    state = problem.initial_state.copy()
    state[:] = np.random.default_rng(3).uniform(0, 2 * math.pi, state.size)
    assert problem.drift is not None
    return problem, state, problem.drift(0.0, state, problem.initial_parameters)
