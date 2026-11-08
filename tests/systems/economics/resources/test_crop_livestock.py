"""Test the `_crop_livestock` module of `skcomplex.systems.economics.resources`."""

import numpy as np

from skcomplex.systems.economics.resources import CropLivestockHouseholdSystem

from ..conftest import _run


def test_crop_livestock_household_buys_livestock_that_joins_three_years_later():
    """Test the STELLA years by hand: purchases join the herd three years later and then bring livestock income."""
    system = CropLivestockHouseholdSystem().prepare().system_
    trajectory = _run(system, 12, observables=CropLivestockHouseholdSystem().get_run_settings()['observables'])
    purchase, livestock = trajectory.values('household.purchase'), trajectory.values('household.livestock')
    assert purchase[1] == 1.0
    assert livestock.tolist()[:6] == [4.0, 4.0, 4.0, 4.0, 4.0, 5.0]
    wealth = trajectory.values('household.wealth')
    expected = 10000.0 + 80000.0 - 75000.0 - 15000.0 * purchase[0]
    assert wealth[1] == expected
    assert np.all(trajectory.values('household.income_from_livestock')[:5] == 0.0)
    assert trajectory.values('household.income_from_livestock')[-4:].mean() > 20000.0
    assert wealth[-3:].max() > 4 * wealth[0]
