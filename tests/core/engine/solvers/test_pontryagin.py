"""Test the optimal control solver by Pontryagin's principle."""

from typing import Annotated

import numpy as np
import pytest
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw, RateLaw
from skcomplex.core.engine.solvers import PontryaginSolver
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System


class QuarticElement(Element):
    """Value steered by a control between -1 and 1."""

    def __init__(self, label: str, *, x: float = 1.0, u: Annotated[float, RealSet.closed(-1, 1)] = 0.0) -> None:
        super().__init__(label)
        self.x = x
        self.u = u


def _quartic(s):
    """Return the rate of the plant and its quartic cost, whose stationarity condition has three roots."""
    plant = s.target
    cost = plant.x**2 + plant.u**4 - plant.u**2
    return [RateLaw(plant.x.diff(s.t), plant.u), ObjectiveLaw(cost, controls=[plant.u])]


def test_pontryagin_matches_riccati_solution(linear_quadratic, riccati_control):
    """Test that the Pontryagin control of the scalar linear-quadratic problem matches the Riccati solution."""
    trajectory = linear_quadratic(PontryaginSolver()).simulate(until=1.0, dt=0.05)
    np.testing.assert_allclose(trajectory.get_values('plant.u'), riccati_control(trajectory.times), atol=1e-3)
    solution = PontryaginSolver(until=1.0).prepare(trajectory.problem).solution_
    assert solution.cost == pytest.approx(np.tanh(1.0) / 2, abs=1e-5)
    np.testing.assert_allclose(solution.costates['plant.x'], -riccati_control(solution.times), atol=1e-6)


def test_pontryagin_terminal_cost_pulls_the_state_to_zero(linear_quadratic):
    """Test that a terminal cost on the state lowers the state at the horizon and enters the cost of the solution."""
    free = linear_quadratic(PontryaginSolver()).simulate(until=1.0, dt=0.1)
    pinned = linear_quadratic(PontryaginSolver(), terminal=10.0).simulate(until=1.0, dt=0.1)
    final = pinned.get_values('plant.x')[-1]
    assert abs(final) < abs(free.get_values('plant.x')[-1])
    running = PontryaginSolver(until=1.0).prepare(pinned.problem).solution_.cost - 10 * final**2
    assert running > PontryaginSolver(until=1.0).prepare(free.problem).solution_.cost


def test_pontryagin_projects_the_control_onto_its_bounds(linear_quadratic):
    """Test that a control whose optimum leaves its bounds is held at the bound."""
    trajectory = linear_quadratic(PontryaginSolver()).simulate(until=1.0, dt=0.1, initial_state={'plant.x': 30.0})
    assert trajectory.get_values('plant.u').min() == pytest.approx(-10.0)


def test_pontryagin_rejects_stationarity_without_unique_solution():
    """Test that a Hamiltonian with several stationary controls is rejected with a pointer to collocation."""
    regulator = Interaction('regulator', roles={'target': 'plant'}, laws=_quartic)
    system = System(elements=[QuarticElement('plant')], interactions=[regulator], solver=PontryaginSolver())
    with pytest.raises(SolverError, match='CollocationSolver'):
        system.prepare().simulate(until=1.0)


def test_pontryagin_reports_a_sweep_that_does_not_converge(linear_quadratic):
    """Test that a sweep that does not converge within its sweeps raises an error naming collocation."""
    with pytest.raises(SolverError, match='did not converge'):
        linear_quadratic(PontryaginSolver(max_iterations=1)).simulate(until=1.0)
