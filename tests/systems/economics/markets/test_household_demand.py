"""Test the `_household_demand` module of `skcomplex.systems.economics.markets`."""

import numpy as np
import pytest

from skcomplex.systems.economics.markets import HouseholdDemandSystem

from ..conftest import _run


def _totals(demand):
    """Return the total demand, expenditure, budgets, and utilities of a run of one price rise of 10 percent."""
    system = HouseholdDemandSystem(demand=demand).prepare().system_
    trajectory = _run(system, 1, observables=HouseholdDemandSystem().get_run_settings()['observables'])
    return trajectory


def test_household_demand_cobb_douglas_keeps_expenditure_and_hurts_the_poorest():
    """Test unit elasticity, constant expenditure, the budget share of food, and the larger utility loss of the poor."""
    trajectory = _totals('cobb-douglas')
    demand, expenditure = trajectory.get_values('total_demand'), trajectory.get_values('total_expenditure')
    budgets = trajectory.get_values('households.budget')[0]
    assert demand[1] / demand[0] == pytest.approx(1 / 1.1)
    assert expenditure[1] == pytest.approx(expenditure[0])
    assert expenditure[0] / budgets.sum() == pytest.approx(0.103, abs=0.005)
    loss = 1 - trajectory.get_values('households.utility')[1] / trajectory.get_values('households.utility')[0]
    poorest, richest = budgets < np.quantile(budgets, 0.2), budgets > np.quantile(budgets, 0.8)
    assert loss[poorest].mean() > loss[richest].mean()


def test_household_demand_elasticities_match_box_3_6():
    """Test that a price rise of 10 percent lowers demand by about 4 percent and raises expenditure by about 5."""
    trajectory = _totals('elasticity')
    demand, expenditure = trajectory.get_values('total_demand'), trajectory.get_values('total_expenditure')
    assert -0.05 < demand[1] / demand[0] - 1 < -0.035
    assert 0.045 < expenditure[1] / expenditure[0] - 1 < 0.06
