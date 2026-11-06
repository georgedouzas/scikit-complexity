"""Test the `OptimalControlSolver` class."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import CollocationSolver, PontryaginSolver
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


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
        Simulation(linear_quadratic, solver=solver()).prepare().run(until=1.0)


def test_optimal_control_rejects_an_unknown_sense(linear_quadratic, quadratic_cost):
    """Test that a sense other than minimum or maximum is rejected."""
    with pytest.raises(ValidationError, match='Unknown sense'):
        Simulation(linear_quadratic, solver=PontryaginSolver(quadratic_cost, sense='best')).prepare().run(
            until=1.0,
        )


def test_optimal_control_needs_bounded_controls(quadratic_cost):
    """Test that a system without a bounded adaptive parameter has nothing to control."""
    with pytest.raises(SolverError, match='no bounded adaptive parameter'):
        (
            Simulation(System(elements=[FreeElement('plant')]), solver=PontryaginSolver(quadratic_cost))
            .prepare()
            .run(until=1.0)
        )


def test_optimal_control_maximises_the_negated_cost(linear_quadratic, quadratic_cost, riccati_control):
    """Test that maximising the negated cost gives the control that minimises the cost."""
    solver = PontryaginSolver(lambda s: -quadratic_cost(s), sense='max')
    trajectory = Simulation(linear_quadratic, solver=solver, dt=0.05).prepare().run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory.values('plant.u'), riccati_control(trajectory.times), atol=1e-3)


def test_optimal_control_stops_at_the_horizon(linear_quadratic, quadratic_cost):
    """Test that a run solved until a horizon cannot be extended beyond it."""
    solver = CollocationSolver(quadratic_cost, nodes=11)
    simulation = Simulation(linear_quadratic, solver=solver, dt=0.5).prepare().run(until=1.0)
    with pytest.raises(SolverError, match='horizon'):
        simulation.extend(until=2.0)


def test_optimal_control_needs_a_finite_horizon(linear_quadratic, quadratic_cost):
    """Test that a run without an end is rejected."""
    with pytest.raises(SolverError, match='finite horizon'):
        Simulation(linear_quadratic, solver=PontryaginSolver(quadratic_cost)).prepare().start()


def test_default_solver_follows_the_laws(linear_quadratic):
    """Test that the default solver keeps the controls of a system at their initial values."""
    trajectory = Simulation(linear_quadratic, dt=0.5).prepare().run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory.values('plant.u'), 0.0)
    np.testing.assert_allclose(trajectory.values('plant.x'), 1.0)
