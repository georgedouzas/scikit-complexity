"""Test the `_adaptivity` module of `skcomplex.models.biology.physiology`."""

from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.physiology import (
    adaptive_response_costs,
)


def test_adaptive_response_costs_fall_with_adaptivity():
    """Test that adaptive response costs less than stimulus-response, avoiding repairs, and anticipation less still."""
    totals, repairs = {}, {}
    for strategy in ('stimulus-response', 'adaptive', 'anticipatory'):
        trajectory = Simulation(adaptive_response_costs(strategy), dt=0.1).run(200).trajectory_
        costs = [trajectory[f'organism.{kind}_cost'][-1] for kind in ('response', 'restore', 'repair', 'prepare')]
        totals[strategy], repairs[strategy] = sum(costs), costs[2]
    assert totals['stimulus-response'] > totals['adaptive'] > totals['anticipatory']
    assert repairs['adaptive'] < 0.01 * repairs['stimulus-response']
