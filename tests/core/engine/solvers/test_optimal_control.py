"""Test the `OptimalControlSolver` class."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import CollocationSolver, PontryaginSolver
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, StepSpec, VisualSpec


class FreeElement(Element):
    """State with an unbounded adaptive parameter."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.u = u

    def _get_laws(self, s):
        """Return the rate of the state."""
        return [RateLaw(s.x, s.u)]


@pytest.mark.parametrize('solver', [PontryaginSolver, CollocationSolver])
def test_optimal_control_needs_an_objective(linear_quadratic, solver):
    """Test that an optimal control solver without an objective raises a named error."""
    with pytest.raises(SolverError, match='has no objective'):
        simulate(linear_quadratic, until=1.0, solver=solver())


def test_optimal_control_rejects_an_unknown_sense(linear_quadratic, quadratic_cost):
    """Test that a sense other than minimum or maximum is rejected."""
    with pytest.raises(ValidationError, match='Unknown sense'):
        simulate(linear_quadratic, until=1.0, solver=PontryaginSolver(quadratic_cost, sense='best'))


def test_optimal_control_needs_bounded_controls(quadratic_cost):
    """Test that a system without a bounded adaptive parameter has nothing to control."""
    with pytest.raises(SolverError, match='no bounded adaptive parameter'):
        simulate(System(elements=[FreeElement('plant')]), until=1.0, solver=PontryaginSolver(quadratic_cost))


def test_optimal_control_maximises_the_negated_cost(linear_quadratic, quadratic_cost, riccati_control):
    """Test that maximising the negated cost gives the control that minimises the cost."""
    solver = PontryaginSolver(lambda s: -quadratic_cost(s), sense='max')
    trajectory = simulate(linear_quadratic, until=1.0, solver=solver, dt=0.05)
    np.testing.assert_allclose(trajectory.values('plant.u'), riccati_control(trajectory.times), atol=1e-3)


def test_optimal_control_stops_at_the_horizon(linear_quadratic, quadratic_cost):
    """Test that a run solved until a horizon cannot be extended beyond it."""
    problem = simulate(linear_quadratic, until=0).problem
    solver = CollocationSolver(quadratic_cost, nodes=11, until=1.0).prepare(problem)
    state = solver.run(StepSpec(0.0, solver.initial_state_, problem.initial_parameters, 1.0)).state_
    with pytest.raises(SolverError, match='horizon'):
        solver.run(StepSpec(1.0, state, problem.initial_parameters, 2.0))


def test_optimal_control_needs_a_finite_horizon(linear_quadratic, quadratic_cost):
    """Test that a run without an end is rejected."""
    problem = simulate(linear_quadratic, until=0).problem
    with pytest.raises(SolverError, match='finite horizon'):
        PontryaginSolver(quadratic_cost).prepare(problem)


def test_default_solver_follows_the_laws(linear_quadratic):
    """Test that the default solver keeps the controls of a system at their initial values."""
    trajectory = simulate(linear_quadratic, until=1.0, dt=0.5)
    np.testing.assert_allclose(trajectory.values('plant.u'), 0.0)
    np.testing.assert_allclose(trajectory.values('plant.x'), 1.0)
