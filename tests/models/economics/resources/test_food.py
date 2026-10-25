"""Test the `_food` module of `skcomplex.models.economics.resources`."""

from skcomplex.models.economics.resources import (
    food_security,
)

from ...conftest import _peak, _run_monitored


def test_food_security_falls_and_recovers_with_policy():
    """Test that self-sufficiency falls as the population grows and that research and extension raise it."""
    baseline = _run_monitored(food_security(), 0.25, 50)
    sufficiency = baseline.values('rice.self_sufficiency')
    assert sufficiency[-1] < 0.4 * sufficiency[0]
    assert 0 < _peak(baseline, 'rice.rice_area') < 10
    withdrawn = _run_monitored(food_security(subsidy=0.0), 0.25, 50).values('rice.self_sufficiency')
    assert withdrawn[-1] < sufficiency[-1]
    policy = food_security(subsidy=0.335, bio_target=0.5, research=1.0, training_target=1.0, intensity_target=2.0)
    assert _run_monitored(policy, 0.25, 50).values('rice.self_sufficiency')[-1] > sufficiency[-1]
