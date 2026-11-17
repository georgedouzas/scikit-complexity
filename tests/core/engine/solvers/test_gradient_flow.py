"""Test the `_gradient_flow` module of `skcomplex.core.engine.solvers`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import GradientFlowSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

from .test_objective import Well


def test_gradient_flow_descends_like_its_exact_solution():
    """Test that the flow of a quadratic well follows `x = c (1 - exp(-2 t))` within the error of its steps."""
    system = System(
        elements=[Element('ball', values={'x': 0.0})],
        interactions=[Well('well')],
        solver=GradientFlowSolver(dt=1e-3),
    )
    trajectory = system.prepare().simulate(until=1.0, dt=0.25)
    assert system.form_ == 'static solution'
    assert trajectory.get_values('ball.x') == pytest.approx(2.0 * (1 - np.exp(-2 * trajectory.times)), abs=1e-2)


def test_gradient_flow_rejects_a_step_that_is_not_positive():
    """Test that a step of zero raises an error naming the step."""
    system = System(
        elements=[Element('ball', values={'x': 0.0})],
        interactions=[Well('well')],
        solver=GradientFlowSolver(dt=0.0),
    )
    with pytest.raises(ValidationError, match='step dt of GradientFlowSolver must be positive'):
        system.prepare().simulate(until=1.0)
