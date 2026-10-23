"""Test the reading of optimal control problems and the following of their solutions over a run."""

import pytest

from skcomplex.core.base import SolverError, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import Rate
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import Collocation, Pontryagin
from skcomplex.core.system import System


class Free(Element):
    """State with an unbounded adaptive parameter and no objective."""

    x: float = state(1.0)
    u: float = parameter(0.0, adaptive=True)

    def laws(self, s):
        """Return the rate of the state."""
        return [Rate(s.x, s.u)]


def test_optimal_control_needs_an_objective():
    """Test that a system without an objective law is rejected by the optimal control solvers."""
    with pytest.raises(SolverError, match='one objective law'):
        Simulation(System(components=[Free('plant')]), solver=Pontryagin()).run(until=1.0)


def test_optimal_control_stops_at_the_horizon(linear_quadratic):
    """Test that a run solved until a horizon cannot be extended beyond it."""
    simulation = Simulation(linear_quadratic, solver=Collocation(nodes=11), dt=0.5).run(until=1.0)
    with pytest.raises(SolverError, match='horizon'):
        simulation.extend(until=2.0)


def test_optimal_control_needs_a_finite_horizon(linear_quadratic):
    """Test that a run without an end is rejected."""
    with pytest.raises(SolverError, match='finite horizon'):
        Simulation(linear_quadratic, solver=Pontryagin()).start()


def test_default_solver_does_not_optimise_objectives(linear_quadratic):
    """Test that the default solver, which follows laws, refuses a system with an objective law."""
    with pytest.raises(SolverError, match='does not optimise'):
        Simulation(linear_quadratic).run(until=1.0)
