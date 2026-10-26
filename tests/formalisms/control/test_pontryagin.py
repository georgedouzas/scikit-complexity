"""Test the optimal control solver by Pontryagin's principle."""

import numpy as np
import pytest

from skcomplex.core.base import SolverError, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import ObjectiveQuantity, RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.control import PontryaginSolver


class QuarticElement(Element):
    """State steered by a control whose cost is quartic, so the stationarity condition has three roots."""

    x: float = state(1.0)
    u: float = parameter(0.0, adaptive=True, bounds=(-1, 1))

    def laws(self, s):
        """Return the rate and the quartic cost."""
        return [RateLaw(s.x, s.u), ObjectiveQuantity('objective', s.x**2 + s.u**4 - s.u**2)]


def test_pontryagin_matches_riccati_solution(linear_quadratic, riccati_control):
    """Test that the Pontryagin control of the scalar linear-quadratic problem matches the Riccati solution."""
    simulation = Simulation(linear_quadratic, solver=PontryaginSolver(), dt=0.05).run(until=1.0)
    trajectory = simulation.trajectory_
    np.testing.assert_allclose(trajectory.values('plant.u'), riccati_control(trajectory.times), atol=1e-3)
    solution = simulation.stepper_.solution_
    assert solution.cost == pytest.approx(np.tanh(1.0) / 2, abs=1e-5)
    np.testing.assert_allclose(solution.costates['plant.x'], -riccati_control(solution.times), atol=1e-6)


def test_pontryagin_records_the_running_cost(linear_quadratic):
    """Test that the trajectory records the running cost of the objective under its label."""
    trajectory = Simulation(linear_quadratic, solver=PontryaginSolver(), dt=0.1).run(until=1.0).trajectory_
    expected = (trajectory.values('plant.x') ** 2 + trajectory.values('plant.u') ** 2) / 2
    np.testing.assert_allclose(trajectory.values('plant.objective'), expected)


def test_pontryagin_projects_the_control_onto_its_bounds(linear_quadratic):
    """Test that a control whose optimum leaves its bounds is held at the bound."""
    simulation = Simulation(linear_quadratic, solver=PontryaginSolver(), dt=0.1, initial_state={'plant.x': 30.0})
    controls = simulation.run(until=1.0).trajectory_.values('plant.u')
    assert controls.min() == pytest.approx(-10.0)


def test_pontryagin_rejects_stationarity_without_unique_solution():
    """Test that a Hamiltonian with several stationary controls is rejected with a pointer to collocation."""
    with pytest.raises(SolverError, match='CollocationSolver'):
        Simulation(System(components=[QuarticElement('plant')]), solver=PontryaginSolver()).run(until=1.0)


def test_pontryagin_reports_a_sweep_that_does_not_converge(linear_quadratic):
    """Test that a sweep that does not converge within its sweeps raises an error naming collocation."""
    with pytest.raises(SolverError, match='did not converge'):
        Simulation(linear_quadratic, solver=PontryaginSolver(max_iterations=1)).run(until=1.0)
