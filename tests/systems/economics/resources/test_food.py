"""Test the `_food` module of `skcomplex.systems.economics.resources`."""

import numpy as np
import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.economics.resources import (
    FoodSecuritySystem,
    PartialAdjustmentInteraction,
)

from ....conftest import NodeElement
from ...conftest import _peak, _run_monitored


def test_food_security_falls_and_recovers_with_policy():
    """Test that self-sufficiency falls as the population grows and that research and extension raise it."""
    baseline = _run_monitored(FoodSecuritySystem().prepare().system_, 0.25, 50, FoodSecuritySystem())
    sufficiency = baseline.get_values('rice.self_sufficiency')
    assert sufficiency[-1] < 0.4 * sufficiency[0]
    assert 0 < _peak(baseline, 'rice.rice_area') < 10
    withdrawn = _run_monitored(
        FoodSecuritySystem(subsidy=0.0).prepare().system_,
        0.25,
        50,
        FoodSecuritySystem(),
    ).get_values(
        'rice.self_sufficiency',
    )
    assert withdrawn[-1] < sufficiency[-1]
    policy = (
        FoodSecuritySystem(subsidy=0.335, bio_target=0.5, research=1.0, training_target=1.0, intensity_target=2.0)
        .prepare()
        .system_
    )
    assert (
        _run_monitored(policy, 0.25, 50, FoodSecuritySystem()).get_values('rice.self_sufficiency')[-1] > sufficiency[-1]
    )


def test_partial_adjustment_follows_a_moving_target():
    """Test that a partial adjustment towards the target `t` follows `t - time (1 - exp(-t / time))`."""
    adjustment = PartialAdjustmentInteraction('adjustment', target='policy', goal=lambda s: s.t, time=2.0)
    system = CompositeSystem(elements=[NodeElement('policy', x=0.0)], interactions=[adjustment])
    trajectory = simulate(system, until=3.0, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=0.5)
    expected = trajectory.times - 2.0 * (1 - np.exp(-trajectory.times / 2.0))
    np.testing.assert_allclose(trajectory.get_values('policy.x'), expected, rtol=1e-7, atol=1e-9)


def test_partial_adjustment_rejects_a_time_that_is_not_a_function():
    """Test that `check` rejects an adjustment time that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='time'):
        PartialAdjustmentInteraction('adjustment', target='policy', time='slow').prepare()
