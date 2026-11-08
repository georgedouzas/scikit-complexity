"""Test the `IMEXEulerSolver` class."""

import numpy as np
import pytest
import skfem

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import (
    IMEXEulerSolver,
)
from skcomplex.systems.physics.heat import HeatLShapeSystem, harmonic


def test_l_shaped_steady_state_converges_at_second_order():
    """Test that heat on the L-shaped mesh converges at second order away from the re-entrant corner."""
    errors = []
    for refine in (2, 3, 4, 5):
        trajectory = simulate(
            HeatLShapeSystem(refine).prepare().system_,
            until=500,
            dt=100,
            solver=IMEXEulerSolver(dt=10.0),
        )
        nodes = skfem.MeshTri.init_lshaped().refined(refine).p.T
        away = np.hypot(nodes[:, 0], nodes[:, 1]) > 0.25
        errors.append(np.abs(trajectory.states[-1] - harmonic(nodes[:, 0], nodes[:, 1]))[away].max())
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert ratios[-1] == pytest.approx(4, rel=0.1)
    assert errors[-1] < 1e-4
