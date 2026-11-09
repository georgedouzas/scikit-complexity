"""Test the `_traps` module of `skcomplex.systems.social.organizations`."""

import numpy as np
import pytest

from skcomplex.core.analysis import loops
from skcomplex.systems.social.organizations import (
    DriftToLowPerformanceSystem,
    EscalationSystem,
    PolicyResistanceSystem,
    ShiftingTheBurdenSystem,
    SuccessToTheSuccessfulSystem,
)

from ...conftest import _run_seeded


def _polarities(system):
    """Return the polarities of the feedback loops of a system at its initial state."""
    return sorted(loops(system).polarities)


def test_success_to_the_successful_gives_everything_to_the_initially_larger():
    """Test that the initially larger competitor takes the whole budget, through the reinforcing loops of the budget."""
    for a, b in ((55.0, 45.0), (45.0, 55.0)):
        trajectory = _run_seeded(SuccessToTheSuccessfulSystem(a=a, b=b).prepare().system_, 100.0, 5.0)
        winner, loser = ('contest.a', 'contest.b') if a > b else ('contest.b', 'contest.a')
        assert trajectory.get_values(winner)[-1] > 99.0
        assert trajectory.get_values(loser)[-1] < 1.0
    assert _polarities(SuccessToTheSuccessfulSystem().prepare().system_) == ['balancing'] * 2 + ['reinforcing'] * 3


def test_shifting_the_burden_lets_the_quick_fix_take_over():
    """Test that the quick fix replaces the capability of the system, which builds up to the goal without it."""
    addicted = _run_seeded(ShiftingTheBurdenSystem().prepare().system_, 40.0, 1.0, ShiftingTheBurdenSystem())
    assert addicted.get_values('burden.capability')[-1] < 0.01
    assert addicted.get_values('burden.intervention')[-1] == pytest.approx(1.0, abs=0.01)
    assert abs(addicted.get_values('burden.symptom')[-1]) < 0.01
    unaided = _run_seeded(ShiftingTheBurdenSystem(fix_rate=0.0).prepare().system_, 40.0, 1.0)
    assert unaided.get_values('burden.capability')[-1] == pytest.approx(1.0, abs=0.01)
    assert 'reinforcing' in _polarities(ShiftingTheBurdenSystem().prepare().system_)


def test_escalation_grows_both_stocks_exponentially():
    """Test that both stocks grow at the rate of the reinforcing loop through the two parties."""
    trajectory = _run_seeded(EscalationSystem().prepare().system_, 100.0, 5.0)
    rate = (-3.0 + np.sqrt(9.0 + 0.8)) / 4.0
    for path in ('race.a', 'race.b'):
        assert np.log(trajectory.get_values(path)[-1] / trajectory.get_values(path)[-5]) / 20.0 == pytest.approx(
            rate,
            rel=1e-3,
        )
    assert 'reinforcing' in _polarities(EscalationSystem().prepare().system_)


def test_policy_resistance_holds_the_stock_where_no_actor_wants_it():
    """Test that the stock settles at the weighted mean of the goals while the efforts escalate in opposite ways."""
    standoff = _run_seeded(PolicyResistanceSystem().prepare().system_, 100.0, 5.0, PolicyResistanceSystem())
    assert standoff.get_values('standoff.stock')[-1] == pytest.approx(50.0, abs=0.1)
    assert standoff.get_values('standoff.compromise')[-1] == pytest.approx(50.0)
    assert standoff.get_values('standoff.effort_a')[-1] > 20.0
    assert standoff.get_values('standoff.effort_b')[-1] < -20.0
    let_up = _run_seeded(PolicyResistanceSystem(strengths=(0.1, 0.0, 0.1)).prepare().system_, 100.0, 5.0)
    assert let_up.get_values('standoff.stock')[-1] == pytest.approx(65.0, abs=0.1)


def test_drift_to_low_performance_erodes_the_goal_unless_it_is_absolute():
    """Test that the performance drifts down with eroding goals, holds with absolute goals, and rises when upbeat."""
    drifting = _run_seeded(DriftToLowPerformanceSystem().prepare().system_, 100.0, 5.0)
    performance = drifting.get_values('performance.performance')
    assert np.all(np.diff(performance) < 0)
    assert performance[-1] < 0.5
    absolute = _run_seeded(DriftToLowPerformanceSystem(absolute=True).prepare().system_, 100.0, 5.0)
    assert absolute.get_values('performance.performance')[-1] == pytest.approx(1.0)
    upbeat = _run_seeded(DriftToLowPerformanceSystem(bias=-0.1).prepare().system_, 100.0, 5.0)
    assert upbeat.get_values('performance.performance')[-1] > 2.0
    assert 'reinforcing' in _polarities(DriftToLowPerformanceSystem().prepare().system_)
