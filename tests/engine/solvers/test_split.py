"""Test the `SplitSolver` class."""

import numpy as np

from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import IMEXEulerSolver
from skcomplex.models.physics.fluids import (
    rayleigh_benard,
)


def test_split_solver_linear_part_matches_linear_rates():
    """Test that the linear part of the rates equals the rates of a small perturbation of the coupled convection."""
    problem = compile_system(rayleigh_benard(1000.0, shape=(16, 16)))
    solver = IMEXEulerSolver().check().prepare(problem)
    solver._build_operator(problem.initial_parameters)
    noise = np.random.default_rng(0).standard_normal((2, 16, 16))
    state = 1e-7 * (noise - np.roll(np.flip(noise, axis=2), 1, axis=2)).ravel()
    rates = problem.drift(0.0, state, problem.initial_parameters)
    assert np.abs(rates - solver._linear(state)).max() < 1e-9 * np.abs(rates).max() + 1e-12
