"""Test the `_fair_food` module of `skcomplex.systems.economics.markets`."""

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import describe
from skcomplex.systems.economics.markets import fair_food


def _fair_share(**kwargs):
    """Return the mean share of fair food over the last steps of a run."""
    trajectory = (
        Simulation(
            built := fair_food(12, 12, **kwargs),
            random_state=0,
            observables=describe('fair_food').observables_of(built),
        )
        .prepare()
        .run(until=25)
        .trajectory_
    )
    return trajectory.values('fair_share')[-10:].mean()


def test_fair_food_ethics_price_and_influence():
    """The fair share rises with the ethics weight, falls with the premium, and rises with social influence."""
    base = _fair_share()
    assert _fair_share(ethics_weight=4.0) > base + 0.05
    assert _fair_share(premium=0.8) < base - 0.05
    assert _fair_share(social_weight=0.0) < base - 0.05
