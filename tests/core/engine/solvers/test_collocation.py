"""Test the optimal control solver by direct collocation."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet
from scipy.optimize import brentq

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw, RateLaw
from skcomplex.core.engine.solvers import CollocationSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

HORIZON = 1.0


class IntegratorElement(Element):
    """Scalar value driven by its bounded control."""

    def __init__(self, label: str, *, x: float = 1.0, u: Annotated[float, RealSet.closed(-10, 10)] = 0.0) -> None:
        super().__init__(label)
        self.x = x
        self.u = u


class FisheryElement(Element):
    """Logistic resource harvested with bounded effort for a discounted profit."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        effort: Annotated[float, RealSet.closed(0, 1)] = 0.0,
        cost: float = 0.2,
        discount: float = 0.1,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.effort = effort
        self.cost = cost
        self.discount = discount


def _regulate(s):
    """Return the rate `x' = u` of the plant and its quadratic running cost `(x^2 + u^2) / 2`."""
    plant = s.target
    return [RateLaw(plant.x.diff(s.t), plant.u), ObjectiveLaw((plant.x**2 + plant.u**2) / 2, controls=[plant.u])]


def _harvest(s):
    """Return the logistic regrowth less the harvest, and minus the discounted profit of the fishery."""
    fishery = s.target
    profit = exp(-fishery.discount * s.t) * (fishery.x - fishery.cost) * fishery.effort
    return [
        RateLaw(fishery.x.diff(s.t), fishery.x * (1 - fishery.x) - fishery.effort * fishery.x),
        ObjectiveLaw(-profit, controls=[fishery.effort]),
    ]


def _linear_quadratic(solver):
    """Return the scalar linear-quadratic problem `x' = u`, `x(0) = 1`, with cost `∫ (x^2 + u^2) / 2 dt`."""
    regulator = Interaction('regulator', roles={'target': 'plant'}, laws=_regulate)
    return System(elements=[IntegratorElement('plant')], interactions=[regulator], solver=solver).prepare()


def _riccati_control(times):
    """Return the optimal control of the scalar linear-quadratic problem, `-sinh(1 - t) / cosh(1)`."""
    return -np.sinh(HORIZON - times) / np.cosh(HORIZON)


def _singular_level(cost, discount):
    """Return the resource level of the singular arc, where the marginal yield net of cost equals the discount."""

    def condition(x):
        return 1 - 2 * x + cost * (1 - x) / (x - cost) - discount

    return brentq(condition, cost + 1e-6, 1.0)


def test_collocation_matches_riccati_solution():
    """Test that the collocation control of the scalar linear-quadratic problem matches the Riccati solution."""
    trajectory = _linear_quadratic(CollocationSolver(nodes=51)).simulate(until=1.0, dt=0.05)
    np.testing.assert_allclose(trajectory.get_values('plant.u'), _riccati_control(trajectory.times), atol=1e-3)


def test_collocation_trapezoidal_matches_riccati_solution_inside_the_run():
    """Test that the trapezoidal control matches the Riccati solution between the end nodes, where it is least exact."""
    solver = CollocationSolver(nodes=101, scheme='trapezoidal')
    trajectory = _linear_quadratic(solver).simulate(until=1.0, dt=0.05)
    inside = slice(1, -1)
    np.testing.assert_allclose(
        trajectory.get_values('plant.u')[inside],
        _riccati_control(trajectory.times[inside]),
        atol=1e-3,
    )


def test_collocation_with_trust_region_method():
    """Test that the trust-region method of SciPy solves the transcription too."""
    solver = CollocationSolver(nodes=11, method='trust-constr', tolerance=1e-8)
    trajectory = _linear_quadratic(solver).simulate(until=1.0, dt=0.1)
    np.testing.assert_allclose(trajectory.get_values('plant.u'), _riccati_control(trajectory.times), atol=1e-2)


def test_collocation_harvests_bang_bang():
    """Test that the optimal effort switches between its bounds and holds the resource at the singular level."""
    solver = CollocationSolver(nodes=101, scheme='trapezoidal')
    harvest = Interaction('harvest', roles={'target': 'fishery'}, laws=_harvest)
    system = System(elements=[FisheryElement('fishery')], interactions=[harvest], solver=solver)
    trajectory = system.prepare().simulate(until=20.0, dt=0.5)
    effort, resource = trajectory.get_values('fishery.effort'), trajectory.get_values('fishery.x')
    level = _singular_level(cost=0.2, discount=0.1)
    assert effort[0] == pytest.approx(1.0, abs=1e-6)
    assert effort[-1] == pytest.approx(1.0, abs=1e-6)
    middle = (trajectory.times > 5) & (trajectory.times < 15)
    np.testing.assert_allclose(resource[middle], level, atol=5e-3)
    np.testing.assert_allclose(effort[middle], 1 - level, atol=1e-2)


def test_collocation_rejects_unknown_scheme():
    """Test that an unknown scheme is rejected when the solver is prepared."""
    with pytest.raises(ValidationError, match='Unknown collocation scheme'):
        _linear_quadratic(CollocationSolver(scheme='euler')).simulate(until=1.0)


def test_collocation_declares_rate_laws():
    """Test that the collocation solver declares that it supports rate laws."""
    assert not CollocationSolver._get_support().get_missing('laws', [RateLaw])
