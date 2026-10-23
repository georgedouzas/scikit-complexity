"""Test the optimal control solver by direct collocation."""

import numpy as np
import pytest
from scipy.optimize import brentq

from skcomplex.core.base import ValidationError
from skcomplex.core.compile import missing_declarations
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import Collocation


def _singular_level(cost, discount):
    """Return the resource level of the singular arc, where the marginal yield net of cost equals the discount."""

    def condition(x):
        return 1 - 2 * x + cost * (1 - x) / (x - cost) - discount

    return brentq(condition, cost + 1e-6, 1.0)


def test_collocation_matches_riccati_solution(linear_quadratic, riccati_control):
    """Test that the collocation control of the scalar linear-quadratic problem matches the Riccati solution."""
    trajectory = Simulation(linear_quadratic, solver=Collocation(nodes=51), dt=0.05).run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory['plant.u'], riccati_control(trajectory.times), atol=1e-3)


def test_collocation_trapezoidal_matches_riccati_solution_inside_the_run(linear_quadratic, riccati_control):
    """Test that the trapezoidal control matches the Riccati solution between the end nodes, where it is least exact."""
    solver = Collocation(nodes=101, scheme='trapezoidal')
    trajectory = Simulation(linear_quadratic, solver=solver, dt=0.05).run(until=1.0).trajectory_
    inside = slice(1, -1)
    np.testing.assert_allclose(trajectory['plant.u'][inside], riccati_control(trajectory.times[inside]), atol=1e-3)


def test_collocation_with_trust_region_method(linear_quadratic, riccati_control):
    """Test that the trust-region method of SciPy solves the transcription too."""
    solver = Collocation(nodes=11, method='trust-constr', tolerance=1e-8)
    trajectory = Simulation(linear_quadratic, solver=solver, dt=0.1).run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory['plant.u'], riccati_control(trajectory.times), atol=1e-2)


def test_collocation_harvests_bang_bang(harvesting):
    """Test that the optimal effort switches between its bounds and holds the resource at the singular level."""
    solver = Collocation(nodes=101, scheme='trapezoidal')
    trajectory = Simulation(harvesting, solver=solver, dt=0.5).run(until=20.0).trajectory_
    effort, resource = trajectory['fishery.effort'], trajectory['fishery.x']
    level = _singular_level(cost=0.2, discount=0.1)
    assert effort[0] == pytest.approx(1.0, abs=1e-6)
    assert effort[-1] == pytest.approx(1.0, abs=1e-6)
    middle = (trajectory.times > 5) & (trajectory.times < 15)
    np.testing.assert_allclose(resource[middle], level, atol=5e-3)
    np.testing.assert_allclose(effort[middle], 1 - level, atol=1e-2)


def test_collocation_rejects_unknown_scheme(linear_quadratic):
    """Test that an unknown scheme is rejected when the solver is prepared."""
    with pytest.raises(ValidationError, match='Unknown collocation scheme'):
        Simulation(linear_quadratic, solver=Collocation(scheme='euler')).run(until=1.0)


def test_optimal_control_solvers_declare_support():
    """Test that the optimal control solvers are registered with their declarations."""
    assert missing_declarations() == []
    assert 'objective' in {str(kind) for kind in Collocation.supports.laws}
