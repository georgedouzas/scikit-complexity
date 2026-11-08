"""Test the `_adaptivity` module of `skcomplex.systems.biology.physiology`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.physiology import (
    AdaptiveResponseCostsSystem,
)


def test_adaptive_response_costs_fall_with_adaptivity():
    """Test that adaptive response costs less than stimulus-response, avoiding repairs, and anticipation less still."""
    totals, repairs = {}, {}
    for strategy in ('stimulus-response', 'adaptive', 'anticipatory'):
        trajectory = simulate(AdaptiveResponseCostsSystem(strategy), dt=0.1, until=200)
        costs = [
            trajectory.values(f'organism.{kind}_cost')[-1] for kind in ('response', 'restore', 'repair', 'prepare')
        ]
        totals[strategy], repairs[strategy] = sum(costs), costs[2]
    assert totals['stimulus-response'] > totals['adaptive'] > totals['anticipatory']
    assert repairs['adaptive'] < 0.01 * repairs['stimulus-response']
