"""Test the `_fair_food` module of `skcomplex.systems.economics.markets`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.markets import FairFoodSystem


def _fair_share(**kwargs):
    """Return the mean share of fair food over the last steps of a run."""
    trajectory = simulate(
        FairFoodSystem(12, 12, **kwargs).prepare().system_,
        until=25,
        random_state=0,
        observables=FairFoodSystem().get_run_settings()['observables'],
    )
    return trajectory.get_values('fair_share')[-10:].mean()


def test_fair_food_ethics_price_and_influence():
    """The fair share rises with the ethics weight, falls with the premium, and rises with social influence."""
    base = _fair_share()
    assert _fair_share(ethics_weight=4.0) > base + 0.05
    assert _fair_share(premium=0.8) < base - 0.05
    assert _fair_share(social_weight=0.0) < base - 0.05
