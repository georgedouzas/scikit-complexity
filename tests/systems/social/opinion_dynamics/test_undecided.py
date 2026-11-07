"""Test the `_undecided` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np
import pytest

from skcomplex.systems.social.opinion_dynamics import ThreeStateOpinionSystem, UndecidedFlowsSystem

from ...conftest import _run_seeded
from ..conftest import _run, _values


def _final(trajectory):
    """Return the final numbers against, for, and undecided."""
    return [trajectory.values(f'population.{name}')[-1] for name in ('against', 'support', 'undecided')]


def test_undecided_flows_settle_at_the_fixed_point_of_constant_rates():
    """Test that constant rates settle at `A : B : C = omega / gamma : omega / eta : 1`."""
    trajectory = _run_seeded(UndecidedFlowsSystem(driven=False).prepare().system_, 200.0, 1.0)
    np.testing.assert_allclose(_final(trajectory), np.array([20.0, 50.0, 1.0]) * 100 / 71, rtol=1e-3)


def test_undecided_drive_leaves_undecided_agents_only_above_the_critical_size():
    """Test that above `N_c = omega (1 / gamma + 1 / eta)` the surplus never decides, and below it everyone does."""
    above = _run_seeded(UndecidedFlowsSystem().prepare().system_, 50.0, 1.0, 'undecided_flows')
    np.testing.assert_allclose(_final(above), [20.0, 50.0, 30.0], rtol=1e-4)
    assert above.values('population.critical_size')[-1] == pytest.approx(70.0)
    below = _run_seeded(UndecidedFlowsSystem(50.0).prepare().system_, 50.0, 1.0)
    against, support, undecided = _final(below)
    assert undecided < 1e-3
    assert against + support == pytest.approx(50.0)


def test_three_state_opinion_orders_only_above_the_critical_sensitivity():
    """Test that the mean opinion stays near zero below `beta_c = 4 / q` and departs from it well above."""
    means = {}
    for sensitivity in (0.5, 8.0):
        trajectory = _run(ThreeStateOpinionSystem(sensitivity=sensitivity), 100)
        means[sensitivity] = np.mean(
            [abs(np.mean(_values(trajectory, 'voters', 'opinion', frame))) for frame in range(50, 101)],
        )
    assert means[0.5] < 0.1
    assert means[8.0] > 3 * means[0.5]


def test_three_state_opinion_draws_half_undecided():
    """Test that half of the draws are undecided, since `f(t) + f(-t) = 1 / 2`."""
    trajectory = _run(ThreeStateOpinionSystem(400, sensitivity=1.0), 20)
    undecided = np.mean([np.mean(_values(trajectory, 'voters', 'opinion', frame) == 0) for frame in range(5, 21)])
    assert undecided == pytest.approx(0.5, abs=0.03)
