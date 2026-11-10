"""Test the `_fair_food_program` module of `skcomplex.systems.economics.markets`."""

import numpy as np

from skcomplex.systems.economics.markets import FairFoodProgramSystem

from ..conftest import _run


def test_fair_food_program_farms_tip_into_the_programme_late():
    """Test that farms wait for most buyers to commit, then join together, and that all needs are met at last."""
    trajectory = _run(FairFoodProgramSystem(), 5000)
    farms, committed = trajectory.get_values('market.member_farms'), trajectory.get_values('market.committed_shops')
    members = trajectory.get_values('market.member_shops')
    first = int(np.argmax(farms > 0))
    assert committed[first] + members[first] > 0.8 * 30
    assert farms[-1] == 20
    assert members[-1] == 30
    assert int(np.argmax(farms == 20)) - int(np.argmax(farms >= 10)) < 100
    assert trajectory.get_values('market.unfulfilled')[-1] <= 30
    assert np.all(np.diff(farms[1:]) >= 0)
