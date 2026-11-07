"""Test the `SplitSolver` class."""

import numpy as np

from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.solvers import IMEXEulerSolver
from skcomplex.systems import create_system
from skcomplex.systems.physics.fluids import RayleighBenardSystem


def test_split_solver_linear_part_matches_linear_rates():
    """Test that the linear part of the rates equals the rates of a small perturbation of the coupled convection."""
    problem = PassCompiler().prepare(RayleighBenardSystem(1000.0, shape=(16, 16)).prepare().system_).run().problem_
    solver = IMEXEulerSolver().prepare(problem)
    solver._build_operator(problem.initial_parameters)
    noise = np.random.default_rng(0).standard_normal((2, 16, 16))
    state = 1e-7 * (noise - np.roll(np.flip(noise, axis=2), 1, axis=2)).ravel()
    rates = problem.drift(0.0, state, problem.initial_parameters)
    assert np.abs(rates - solver._linear(state)).max() < 1e-9 * np.abs(rates).max() + 1e-12


def test_split_solver_keeps_the_rates_of_a_linear_program_in_the_remainder():
    """Test that a solver prepares on linear programs and that only diffusion enters the linear part of their rates."""
    problem = PassCompiler().prepare(create_system('comets', shape=(3, 3))).run().problem_
    solver = IMEXEulerSolver().prepare(problem)
    solver._build_operator(problem.initial_parameters)
    assert abs(solver._linear(problem.initial_state).sum()) < 1e-9
    assert np.abs(solver._remainder(0.0, problem.initial_state, problem.initial_parameters)).max() > 0.0
