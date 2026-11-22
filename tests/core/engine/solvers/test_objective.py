"""Test the `_objective` module of `skcomplex.core.engine.solvers`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, ObjectiveLaw
from skcomplex.core.engine.solvers import GradientFlowSolver, MetropolisSolver, MinimizeSolver
from skcomplex.core.engine.solvers._objective import BaseObjectiveSolver
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Well(Interaction):
    """Interaction whose objective is a quadratic well around a centre."""

    def __init__(self, label: str, *, target: Role = 'ball', centre: float = 2.0) -> None:
        super().__init__(label)
        self.target = target
        self.centre = centre

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the well as the objective."""
        super()._prepare(context)
        x = context.target.x
        self.laws_: list[BaseLaw] = [ObjectiveLaw((x - context.centre) ** 2, x)]


@pytest.mark.parametrize('solver', [MinimizeSolver, GradientFlowSolver, MetropolisSolver])
def test_objective_solvers_read_the_objective_and_its_gradient(solver):
    """Test that every objective solver gives the value and the gradient of the objective at a state."""
    assert issubclass(solver, BaseObjectiveSolver)
    system = System(elements=[Element('ball', values={'x': 0.0})], interactions=[Well('well')], solver=solver())
    prepared = system.prepare().solver_
    prepared.prepare(system._problem)
    value, gradient = prepared._get_objective_function(system._problem)(
        np.array([0.0]),
        system._problem.initial_parameters,
    )
    assert (value, gradient.tolist()) == (pytest.approx(4.0), [pytest.approx(-4.0)])
