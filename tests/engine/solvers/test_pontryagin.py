"""Test the optimal control solver by Pontryagin's principle."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import PontryaginSolver
from skcomplex.errors import SolverError
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec


class QuarticElement(Element):
    """State steered by a control whose cost is quartic, so the stationarity condition has three roots."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.u = u

    def _get_laws(self, s):
        """Return the rate."""
        return [RateLaw(s.x, s.u)]


def _quartic_cost(s):
    """Return the quartic cost of the plant."""
    return s.plant.x**2 + s.plant.u**4 - s.plant.u**2


def test_pontryagin_matches_riccati_solution(linear_quadratic, quadratic_cost, riccati_control):
    """Test that the Pontryagin control of the scalar linear-quadratic problem matches the Riccati solution."""
    simulation = Simulation(linear_quadratic, solver=PontryaginSolver(quadratic_cost), dt=0.05).check().run(until=1.0)
    trajectory = simulation.trajectory_
    np.testing.assert_allclose(trajectory.values('plant.u'), riccati_control(trajectory.times), atol=1e-3)
    solution = simulation.stepper_.solution_
    assert solution.cost == pytest.approx(np.tanh(1.0) / 2, abs=1e-5)
    np.testing.assert_allclose(solution.costates['plant.x'], -riccati_control(solution.times), atol=1e-6)


def test_pontryagin_terminal_cost_pulls_the_state_to_zero(linear_quadratic, quadratic_cost):
    """Test that a terminal cost on the state lowers the state at the horizon and enters the cost of the solution."""
    free = Simulation(linear_quadratic, solver=PontryaginSolver(quadratic_cost), dt=0.1).check().run(until=1.0)
    solver = PontryaginSolver(quadratic_cost, terminal=lambda s: 10 * s.plant.x**2)
    pinned = Simulation(linear_quadratic, solver=solver, dt=0.1).check().run(until=1.0)
    final = pinned.trajectory_.values('plant.x')[-1]
    assert abs(final) < abs(free.trajectory_.values('plant.x')[-1])
    running = pinned.stepper_.solution_.cost - 10 * final**2
    assert running > free.stepper_.solution_.cost


def test_pontryagin_projects_the_control_onto_its_bounds(linear_quadratic, quadratic_cost):
    """Test that a control whose optimum leaves its bounds is held at the bound."""
    solver = PontryaginSolver(quadratic_cost)
    simulation = Simulation(linear_quadratic, solver=solver, dt=0.1, initial_state={'plant.x': 30.0}).check()
    controls = simulation.run(until=1.0).trajectory_.values('plant.u')
    assert controls.min() == pytest.approx(-10.0)


def test_pontryagin_rejects_stationarity_without_unique_solution():
    """Test that a Hamiltonian with several stationary controls is rejected with a pointer to collocation."""
    with pytest.raises(SolverError, match='CollocationSolver'):
        Simulation(System(elements=[QuarticElement('plant')]), solver=PontryaginSolver(_quartic_cost)).check().run(
            until=1.0,
        )


def test_pontryagin_reports_a_sweep_that_does_not_converge(linear_quadratic, quadratic_cost):
    """Test that a sweep that does not converge within its sweeps raises an error naming collocation."""
    with pytest.raises(SolverError, match='did not converge'):
        Simulation(linear_quadratic, solver=PontryaginSolver(quadratic_cost, max_iterations=1)).check().run(until=1.0)
