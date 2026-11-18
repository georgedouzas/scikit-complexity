"""Test the `OptimalControlSolver` class."""

from typing import Annotated

import numpy as np
import pytest
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw, RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import CollocationSolver, PontryaginSolver, StepSpec
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.systems import System


class FreeElement(Element):
    """State with an unbounded adaptive parameter."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(label)
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
        simulate(CompositeSystem(elements=[FreeElement('plant')]), until=1.0, solver=PontryaginSolver(quadratic_cost))


def test_optimal_control_maximises_the_negated_cost(linear_quadratic, quadratic_cost, riccati_control):
    """Test that maximising the negated cost gives the control that minimises the cost."""
    solver = PontryaginSolver(lambda s: -quadratic_cost(s), sense='max')
    trajectory = simulate(linear_quadratic, until=1.0, solver=solver, dt=0.05)
    np.testing.assert_allclose(trajectory.get_values('plant.u'), riccati_control(trajectory.times), atol=1e-3)


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


class CappedPlantElement(Element):
    """A plant whose control lies between -0.2 and 0."""

    def __init__(self, label: str, x: float = 1.0, u: Annotated[float, RealSet.closed(-0.2, 0.0)] = 0.0) -> None:
        super().__init__(label)
        self.x = x
        self.u = u


def _regulated_plant(solver='auto'):
    """Return the linear-quadratic problem written with an objective law that names its control."""
    plant = Element('plant', values={'x': 1.0, 'u': 0.0})
    regulator = Interaction(
        'regulator',
        roles={'target': 'plant'},
        laws=lambda s: [
            RateLaw(s.target.x.diff(s.t), s.target.u),
            ObjectiveLaw((s.target.x**2 + s.target.u**2) / 2, controls=[s.target.u]),
        ],
    )
    return System(elements=[plant], interactions=[regulator], solver=solver).prepare()


@pytest.mark.parametrize('solver', [PontryaginSolver(), CollocationSolver(nodes=51), 'auto'])
def test_optimal_control_reads_the_controls_of_an_objective_law(solver, riccati_control):
    """Test that an objective law with controls gives the Riccati control, with the default solver too."""
    trajectory = _regulated_plant(solver).simulate(until=1.0, dt=0.05)
    np.testing.assert_allclose(trajectory.get_values('plant.u'), riccati_control(trajectory.times), atol=2e-3)


def test_optimal_control_bounds_a_control_by_its_allowed_values():
    """Test that the allowed values of a control bound it, as a budget caps a rate of vaccination."""
    plant = CappedPlantElement('plant')
    regulator = Interaction(
        'regulator',
        roles={'target': 'plant'},
        laws=lambda s: [
            RateLaw(s.target.x.diff(s.t), s.target.u),
            ObjectiveLaw((s.target.x**2 + s.target.u**2) / 2, controls=[s.target.u]),
        ],
    )
    system = System(elements=[plant], interactions=[regulator], solver=CollocationSolver(nodes=21)).prepare()
    controls = system.simulate(until=1.0, dt=0.1).get_values('plant.u')
    assert controls.min() >= -0.2 - 1e-8
    np.testing.assert_allclose(controls[:3], -0.2, atol=1e-6)


def test_default_solver_follows_the_laws(linear_quadratic):
    """Test that the default solver keeps the controls of a system at their initial values."""
    trajectory = simulate(linear_quadratic, until=1.0, dt=0.5)
    np.testing.assert_allclose(trajectory.get_values('plant.u'), 0.0)
    np.testing.assert_allclose(trajectory.get_values('plant.x'), 1.0)
