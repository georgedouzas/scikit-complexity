"""Test the `_scenario_spec` module of `skcomplex.core.components.specs`."""

import dataclasses

import numpy as np
import pytest

from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.experiments import compare_scenarios
from skcomplex.core.specs import BaseSpec
from skcomplex.systems.economics.resources import HarvestingSystem


def _harvesting() -> CompositeSystem:
    """Return the parts of the harvesting system, the factory of the scenario experiments."""
    return HarvestingSystem().prepare().system_


def test_scenario_spec_changes_a_parameter_at_its_scheduled_time():
    """Test that halving the effort at time 50 changes the stock only after time 50, with the same seeds."""
    scenarios = [ScenarioSpec('baseline'), ScenarioSpec('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
    result = compare_scenarios(_harvesting, scenarios, until=80, outputs=['land/stock.n'], random_state=0)
    baseline, quota = result.get_series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 50], quota[times <= 50])
    assert quota[-1] > baseline[-1]
    assert result.runs[0].seed == result.runs[1].seed


def test_scenario_spec_keeps_the_initial_value_before_its_first_change():
    """Test that a parameter keeps its initial value before its first change and the last due change after it."""
    schedule = [(50.0, 'harvest.effort', 0.025), (20.0, 'harvest.effort', 0.04)]
    scenarios = [ScenarioSpec('baseline'), ScenarioSpec('late', schedule=schedule)]
    result = compare_scenarios(_harvesting, scenarios, until=80, outputs=['land/stock.n'], random_state=0)
    baseline, late = result.get_series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 20], late[times <= 20])
    assert not np.array_equal(baseline[times > 21], late[times > 21])


def test_scenario_spec_is_a_frozen_spec():
    """Test that a scenario is a frozen spec whose default is the baseline."""
    scenario = ScenarioSpec()
    assert isinstance(scenario, BaseSpec)
    assert scenario.name == 'baseline'
    with pytest.raises(dataclasses.FrozenInstanceError):
        scenario.name = 'quota'
