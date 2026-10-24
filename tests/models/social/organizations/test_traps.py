"""Test the `_traps` module of `skcomplex.models.social.organizations`."""

import numpy as np
import pytest

from skcomplex.core.analysis import loops
from skcomplex.models.social.organizations import escalation, shifting_the_burden, success_to_the_successful

from ...conftest import _run_seeded


def _polarities(system):
    """Return the polarities of the feedback loops of a system at its initial state."""
    return sorted(loop.polarity for loop in loops(system).loops)


def test_success_to_the_successful_gives_everything_to_the_initially_larger():
    """Test that the initially larger competitor takes the whole budget, through two reinforcing loops."""
    for a, b in ((55.0, 45.0), (45.0, 55.0)):
        trajectory = _run_seeded(success_to_the_successful(a=a, b=b), 100.0, 5.0)
        winner, loser = ('contest.a', 'contest.b') if a > b else ('contest.b', 'contest.a')
        assert trajectory[winner][-1] > 99.0
        assert trajectory[loser][-1] < 1.0
    assert _polarities(success_to_the_successful()) == ['balancing', 'balancing', 'reinforcing', 'reinforcing']


def test_shifting_the_burden_lets_the_quick_fix_take_over():
    """Test that the quick fix replaces the capability of the system, which builds up to the goal without it."""
    addicted = _run_seeded(shifting_the_burden(), 40.0, 1.0)
    assert addicted['burden.capability'][-1] < 0.01
    assert addicted['burden.intervention'][-1] == pytest.approx(1.0, abs=0.01)
    assert abs(addicted['burden.symptom'][-1]) < 0.01
    unaided = _run_seeded(shifting_the_burden(fix_rate=0.0), 40.0, 1.0)
    assert unaided['burden.capability'][-1] == pytest.approx(1.0, abs=0.01)
    assert 'reinforcing' in _polarities(shifting_the_burden())


def test_escalation_grows_both_stocks_exponentially():
    """Test that both stocks grow at the rate of the reinforcing loop through the two parties."""
    trajectory = _run_seeded(escalation(), 100.0, 5.0)
    rate = (-3.0 + np.sqrt(9.0 + 0.8)) / 4.0
    for path in ('race.a', 'race.b'):
        assert np.log(trajectory[path][-1] / trajectory[path][-5]) / 20.0 == pytest.approx(rate, rel=1e-3)
    assert 'reinforcing' in _polarities(escalation())
